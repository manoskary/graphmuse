import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_sparse import coalesce
# TODO replace torch geometric by local implementation
from torch_geometric.utils import softmax
from torch_geometric.nn import SAGEConv, GraphConv
from collections import namedtuple

class Encoder(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation=F.relu, dropout=0.1):
        super(Encoder, self).__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(self.in_feats, self.n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(self.n_hidden, self.n_hidden))

    def forward(self, edge_index, inputs):
        h = inputs
        for l, conv in enumerate(self.layers):
            h = conv(edge_index, h)
            h = self.activation(F.normalize(h))
            h = self.dropout(h)
        return h


class EdgePooling(torch.nn.Module):
    r"""The edge pooling operator from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`_ and
    `"Edge Contraction Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/1905.10990>`_ papers.
    In short, a score is computed for each edge.
    Edges are contracted iteratively according to that score unless one of
    their nodes has already been part of a contracted edge.
    To duplicate the configuration from the "Towards Graph Pooling by Edge
    Contraction" paper, use either
    :func:`EdgePooling.compute_edge_score_softmax`
    or :func:`EdgePooling.compute_edge_score_tanh`, and set
    :obj:`add_to_edge_score` to :obj:`0`.
    To duplicate the configuration from the "Edge Contraction Pooling for
    Graph Neural Networks" paper, set :obj:`dropout` to :obj:`0.2`.
    Args:
        in_channels (int): Size of each input sample.
        edge_score_method (function, optional): The function to apply
            to compute the edge score from raw edge scores. By default,
            this is the softmax over all incoming edges for each node.
            This function takes in a :obj:`raw_edge_score` tensor of shape
            :obj:`[num_nodes]`, an :obj:`edge_index` tensor and the number of
            nodes :obj:`num_nodes`, and produces a new tensor of the same size
            as :obj:`raw_edge_score` describing normalized edge scores.
            Included functions are
            :func:`EdgePooling.compute_edge_score_softmax`,
            :func:`EdgePooling.compute_edge_score_tanh`, and
            :func:`EdgePooling.compute_edge_score_sigmoid`.
            (default: :func:`EdgePooling.compute_edge_score_softmax`)
        dropout (float, optional): The probability with
            which to drop edge scores during training. (default: :obj:`0`)
        add_to_edge_score (float, optional): This is added to each
            computed edge score. Adding this greatly helps with unpool
            stability. (default: :obj:`0.5`)
    """

    unpool_description = namedtuple(
        "UnpoolDescription",
        ["edge_index", "cluster", "batch", "new_edge_score"])

    def __init__(self, in_channels, edge_score_method=None, dropout=0,
                 add_to_edge_score=0.5):
        super(EdgePooling, self).__init__()
        self.in_channels = in_channels
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_softmax
        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.dropout = dropout

        self.lin = torch.nn.Linear(2 * in_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    @staticmethod
    def compute_edge_score_softmax(raw_edge_score, edge_index, num_nodes):
        return softmax(raw_edge_score, edge_index[1], num_nodes=num_nodes)

    @staticmethod
    def compute_edge_score_tanh(raw_edge_score, edge_index, num_nodes):
        return torch.tanh(raw_edge_score)

    @staticmethod
    def compute_edge_score_sigmoid(raw_edge_score, edge_index, num_nodes):
        return torch.sigmoid(raw_edge_score)

    def forward(self, x, edge_index, batch):
        r"""Forward computation which computes the raw edge score, normalizes
        it, and merges the edges.
        Args:
            x (Tensor): The node features.
            edge_index (LongTensor): The edge indices.
            batch (LongTensor): Batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.
        Return types:
            * **x** *(Tensor)* - The pooled node features.
            * **edge_index** *(LongTensor)* - The coarsened edge indices.
            * **batch** *(LongTensor)* - The coarsened batch vector.
            * **unpool_info** *(unpool_description)* - Information that is
              consumed by :func:`EdgePooling.unpool` for unpooling.
        """
        # Run nodes on each edge through a linear layer
        e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        e = self.lin(e).view(-1)
        e = F.dropout(e, p=self.dropout, training=self.training)
        e = self.compute_edge_score(e, edge_index, x.size(0))
        e = e + self.add_to_edge_score
        x, edge_index, batch, unpool_info = self.__merge_edges__(
            x, edge_index, batch, e)

        return x, edge_index.long(), batch, unpool_info

    def __merge_edges__(self, x, edge_index, batch, edge_score):
        nodes_remaining = set(range(x.size(0)))

        cluster = torch.empty_like(batch, device=torch.device('cpu'))
        # edge_argsort = torch.argsort(edge_score, descending=True)
        edge_argsort = edge_score.detach().cpu().numpy().argsort(kind='stable')[::-1]  # Use stable sort

        # Iterate through all edges, selecting it if it is not incident to
        # another already chosen edge.
        i = 0
        new_edge_indices = []
        edge_index_cpu = edge_index.cpu()
        for edge_idx in edge_argsort.tolist():
            source = edge_index_cpu[0, edge_idx].item()
            if source not in nodes_remaining:
                continue

            target = edge_index_cpu[1, edge_idx].item()
            if target not in nodes_remaining:
                continue

            new_edge_indices.append(edge_idx)

            cluster[source] = i
            nodes_remaining.remove(source)

            if source != target:
                cluster[target] = i
                nodes_remaining.remove(target)

            i += 1

        # The remaining nodes are simply kept.
        for node_idx in nodes_remaining:
            cluster[node_idx] = i
            i += 1
        cluster = cluster.to(x.device)

        # We compute the new features as an addition of the old ones.
        new_x = scatter_add(x, cluster, dim=0, dim_size=i)
        new_edge_score = edge_score[new_edge_indices]
        if len(nodes_remaining) > 0:
            remaining_score = x.new_ones(
                (new_x.size(0) - len(new_edge_indices), ))
            new_edge_score = torch.cat([new_edge_score, remaining_score])
        new_x = new_x * new_edge_score.view(-1, 1)

        N = new_x.size(0)
        new_edge_index, _ = coalesce(cluster[edge_index], None, N, N)

        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        unpool_info = self.unpool_description(edge_index=edge_index,
                                              cluster=cluster, batch=batch,
                                              new_edge_score=new_edge_score)

        return new_x, new_edge_index, new_batch, unpool_info

    def unpool(self, x, unpool_info):
        r"""Unpools a previous edge pooling step.
        For unpooling, :obj:`x` should be of same shape as those produced by
        this layer's :func:`forward` function. Then, it will produce an
        unpooled :obj:`x` in addition to :obj:`edge_index` and :obj:`batch`.
        Args:
            x (Tensor): The node features.
            unpool_info (unpool_description): Information that has
                been produced by :func:`EdgePooling.forward`.
        Return types:
            * **x** *(Tensor)* - The unpooled node features.
            * **edge_index** *(LongTensor)* - The new edge indices.
            * **batch** *(LongTensor)* - The new batch vector.
        """
        new_x = x / unpool_info.new_edge_score.view(-1, 1)
        new_x = new_x[unpool_info.cluster]
        return new_x, unpool_info.edge_index, unpool_info.batch

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.in_channels)

class HGEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, depth,
                 dropout_ratio, activation=F.relu):
        super(HGEncoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.depth = depth
        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)
        self.gcn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.mu_layer = nn.Linear(self.hidden_channels, self.hidden_channels)
        self.log_var_layer = nn.Linear(self.hidden_channels, self.hidden_channels)
        self.gcn_layers.append(GraphConv(in_channels, self.hidden_channels))
        for layer in range(depth):
            self.gcn_layers.append(GraphConv(self.hidden_channels, self.hidden_channels))
            self.pool_layers.append(EdgePooling(self.hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mu_layer.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.log_var_layer.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x, edge_index, batch):
        unpool_infos = []
        h = x
        for i, conv in enumerate(self.gcn_layers):
            h = conv(h, edge_index)
            h = F.normalize(h)
            h = self.activation(h)
            h = self.dropout(h)
            if i != len(self.gcn_layers) - 1:
                h, edge_index, batch, unpool_info = self.pool_layers[i](h, edge_index, batch)
                unpool_infos += [unpool_info]
        mu = self.mu_layer(h)
        log_var = self.log_var_layer(h)
        return h, mu, log_var, unpool_infos


class HGDecoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, depth,
                 dropout_ratio, activation=F.relu):
        super(HGDecoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)
        self.gcn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.gcn_layers.append(GraphConv(self.hidden_channels, out_channels))
        for layer in range(depth):
            self.gcn_layers.append(GraphConv(self.hidden_channels, self.hidden_channels))

    def unpool(self, x, unpool_info):
        new_x = x / unpool_info.new_edge_score.view(-1, 1)
        new_x = new_x[unpool_info.cluster]
        return new_x, unpool_info.edge_index, unpool_info.batch

    def forward(self, x, edge_index, unpool_infos):
        h = x
        for i, conv in reversed(list(enumerate(self.gcn_layers))):
            if i != 0:
                h, edge_index, batch = self.unpool(h, unpool_infos[i-1])
                h = conv(h, edge_index)
                h = F.normalize(h)
                h = self.activation(h)
                h = self.dropout(h)
            else:
                h = conv(h, edge_index)
        return h

class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, recon_x, x, mean, log_var):
        if recon_x.shape[1] != x.shape[1]:
            x = x[:, :recon_x.shape[1]]
        recon_loss = self.mse(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return torch.add(recon_loss, kl_loss)


class HGVAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth, dropout_ratio, activation=F.relu):
        super(HGVAE, self).__init__()
        self.encoder = HGEncoder(in_channels, hidden_channels, depth, dropout_ratio, activation)
        self.decoder = HGDecoder(hidden_channels, out_channels, depth, dropout_ratio, activation)

    def sampling(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.rand_like(std)
        return torch.mul(eps, torch.add(std, mean))

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        h, mu, log_var, unpool_infos = self.encoder(x, edge_index, batch)
        z = self.sampling(mu, log_var)
        h = self.decoder(z, edge_index, unpool_infos) # sampled z causes gradient computation to fail.
        return h, mu, log_var



