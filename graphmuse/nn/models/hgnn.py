import torch.nn as nn
import torch
import torch.nn.functional as F


def h_add(u, v, eps=1e-5):
    "add two tensors in hyperbolic space."
    v = v + eps
    th_dot_u_v = 2. * torch.sum(u * v, dim=1, keepdim=True)
    th_norm_u_sq = torch.sum(u * u, dim=1, keepdim=True)
    th_norm_v_sq = torch.sum(v * v, dim=1, keepdim=True)
    denominator = 1. + th_dot_u_v + th_norm_v_sq * th_norm_u_sq
    result = (1. + th_dot_u_v + th_norm_v_sq) / (denominator + eps) * u + \
             (1. - th_norm_u_sq) / (denominator + eps) * v
    return torch.renorm(result, 2, 0, (1-eps))


def exp_map_zero(v, eps=1e-5):
    """
    Exp map from tangent space of zero to hyperbolic space
    Args:
        v: [batch_size, *] in tangent space
    """
    v = v + eps
    norm_v = torch.norm(v, 2, 1, keepdim=True)
    result = F.tanh(norm_v) / (norm_v) * v
    return torch.renorm(result, 2, 0, (1-eps))


def log_map_zero(v, eps=1e-5):
    """
        Exp map from hyperbolic space of zero to tangent space
        Args:
            v: [batch_size, *] in hyperbolic space
    """
    diff = v + eps
    norm_diff = torch.norm(v, 2, 1, keepdim=True)
    atanh = torch.min(norm_diff, torch.Tensor([1.0 - eps]).to(v.get_device()))
    return 1. / atanh / norm_diff * diff


def h_mul(u, v):
    out = torch.mm(u, log_map_zero(v))
    out = exp_map_zero(out)
    return out


class CentroidDistance(nn.Module):
    """
    Implement a model that calculates the pairwise distances between node representations
    and centroids
    """
    def __init__(self, input_channels, output_channels):
        super(CentroidDistance, self).__init__()
        # centroid embedding
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.centroid_embedding = nn.Embedding(input_channels, output_channels,
            sparse=False,
            scale_grad_by_freq=False)
        self.init_embed()

    def init_embed(self, irange=1e-2):
        self.centroid_embeddin.weight.data.uniform_(-irange, irange)
        self.centroid_embeddin.weight.data.copy_(self.normalize(self.centroid_embeddin.weight.data))

    def forward(self, node_repr, mask):
        """
        Args:
            node_repr: [node_num, embed_size]
            mask: [node_num, 1] 1 denote real node, 0 padded node
        return:
            graph_centroid_dist: [1, num_centroid]
            node_centroid_dist: [1, node_num, num_centroid]
        """
        node_num = node_repr.size(0)

        # broadcast and reshape node_repr to [node_num * num_centroid, embed_size]
        node_repr = node_repr.unsqueeze(1).expand(-1, self.input_channels, -1).contiguous().view(-1, self.output_channels)

        # broadcast and reshape centroid embeddings to [node_num * num_centroid, embed_size]
        centroid_repr = self.centroid_embedding(torch.arange(self.input_channels).cuda())
        centroid_repr = centroid_repr.unsqueeze(0).expand(node_num, -1,-1).contiguous().view(-1, self.output_channels)
        # get distance
        node_centroid_dist = self.manifold.distance(node_repr, centroid_repr)
        node_centroid_dist = node_centroid_dist.view(1, node_num, self.output_channels) * mask
        # average pooling over nodes
        graph_centroid_dist = torch.sum(node_centroid_dist, dim=1) / torch.sum(mask)
        return graph_centroid_dist, node_centroid_dist


class RiemannianSGD(torch.optim.optimizer.Optimizer):
    """Riemannian stochastic gradient descent.
    Args:
        model_params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """

    def __init__(self,model_params, lr):
        defaults = dict(lr=lr)
        self.tanh = nn.Tanh()
        super(RiemannianSGD, self).__init__(model_params, defaults)

    def step(self, lr=None):
        """
        Performs a single optimization step.
        Arguments:
            lr (float, optional): learning rate for the current update.
        """
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                d_p = self.rgrad(p, d_p)
                if lr is None:
                    lr = group['lr']
                p.data = torch.renorm(self.exp_map_x(p, -lr * d_p), 2, 0, (1. - 1e-5))
        return loss

    def exp_map_x(self, x, v):
        """
        Exp map from tangent space of x to hyperbolic space
        """
        v = v + 1e-5 # Perturbe v to avoid dealing with v = 0
        norm_v = torch.norm(v, 2, 1, keepdim=True)
        lambda_x = 2. / (1 - torch.sum(x * x, dim=1, keepdim=True))
        second_term = (self.tanh(lambda_x * norm_v / 2) / norm_v) * v
        return torch.renorm(h_add(x, second_term), 2, 0, (1. - 1e-5))

    def rgrad(self, p, d_p):
        """
        Function to compute Riemannian gradient from the
        Euclidean gradient in the Poincare ball.
        Args:
            p (Tensor): Current point in the ball
            d_p (Tensor): Euclidean gradient at p
        """
        p_sqnorm = torch.sum(p.data ** 2, dim=-1, keepdim=True)
        d_p = d_p * ((1 - p_sqnorm) ** 2 / 4.0).expand_as(d_p)
        return d_p


class HyperbolicSageConv(nn.Module):
    def __init__(self, input_channels, output_channels, bias=True):
        super(HyperbolicSageConv, self).__init__()
        self.bias = bias
        self.embed = torch.zeros((input_channels, input_channels), requires_grad=True)
        self.layer = torch.zeros((2*input_channels, output_channels), requires_grad=True)
        self.reset_parameters()
        self.embed = nn.Parameter(self.embed)
        self.layer = nn.Parameter(self.layer)
        if self.bias:
            self.embed_bias = torch.zeros((input_channels, 1), requires_grad=True)
            self.layer_bias = torch.zeros((output_channels, 1), requires_grad=True)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embed, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.layer, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.constant(self.embed_bias, 0.)
            nn.init.constant(self.layer_bias, 0.)

    def forward(self, x, adj):
        h = h_mul(self.embed_bias, x)
        if self.bias:
            h = h_add(h, self.embed_bias)
        neigh = exp_map_zero(torch.mm(adj, log_map_zero(h)).sum(dim=1))
        h = h_mul(self.layer, torch.cat((x, neigh)))
        if self.bias:
            h = h_add(h, self.layer_bias)
        return h


class HGNN(nn.Module):
    """
    Hyperbolic Graph Neural Network (HGNN) implementation.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels.
    output_channels : int
        Number of output channels.
    num_layers : int
        Number of layers in the HGNN.
    activation : callable, optional
        Activation function, by default F.relu.
    dropout : float, optional
        Dropout rate, by default 0.5.

    Examples
    --------
    >>> hgnn = HGNN(input_channels=16, hidden_channels=32, output_channels=8, num_layers=3)
    >>> x = torch.randn(10, 16)
    >>> adj = torch.randint(0, 2, (10, 10))
    >>> out = hgnn(x, adj)
    >>> print(out.shape)
    torch.Size([10, 8])
    """
    def __init__(self, input_channels, hidden_channels, output_channels, num_layers, activation=F.relu, dropout=0.5):
        super(HGNN, self).__init__()
        self.init_layer = exp_map_zero
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(HyperbolicSageConv(input_channels, hidden_channels))
        for i in range(num_layers - 1):
            self.layers.append(HyperbolicSageConv(hidden_channels, hidden_channels))
        self.layers.append(HyperbolicSageConv(hidden_channels, output_channels))

    def forward(self, x, adj):
        """
        Forward pass for the Hyperbolic Graph Neural Network (HGNN).

        Parameters
        ----------
        x : torch.Tensor
            Input node features.
        adj : torch.Tensor
            Adjacency matrix.

        Returns
        -------
        torch.Tensor
            Output node features.
        """
        # 0-projection to Poincare Hyperbolic space.
        h = self.init_layer(x)
        for i, layer in enumerate(self.layers):
            h = layer(x, adj)
            if i != len(layer)-1:
                h = exp_map_zero(self.activation(log_map_zero(h)))
                h = self.dropout(h)
        return h

class HGNN_NODE(nn.Module):
    """
    HGNN_NODE is a node classification model using Hyperbolic Graph Neural Network (HGNN).

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels.
    output_channels : int
        Number of output channels.
    num_layers : int
        Number of layers in the HGNN.
    activation : callable, optional
        Activation function, by default F.relu.
    dropout : float, optional
        Dropout rate, by default 0.5.

    Examples
    --------
    >>> hgnn_node = HGNN_NODE(input_channels=16, hidden_channels=32, output_channels=8, num_layers=3)
    >>> adj = torch.randint(0, 2, (10, 10))
    >>> features = torch.randn(10, 16)
    >>> out = hgnn_node(adj, features)
    >>> print(out.shape)
    torch.Size([10, 8])
    """
    def __init__(self, input_channels, hidden_channels, output_channels, num_layers, activation=F.relu, dropout=0.5):
        self.input_embedding  = nn.Linear(input_channels, hidden_channels)
        self.hgnn = HGNN(hidden_channels, hidden_channels, hidden_channels, num_layers, activation, dropout)
        self.distance = CentroidDistance(hidden_channels, hidden_channels)
        self.output_linear = nn.Linear(hidden_channels, hidden_channels)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.input_embedding.weight, gain=gain)
        nn.init.xavier_uniform_(self.output_linear.weight, gain=gain)

    def forward(self, adj, features):
        """
        Forward pass for the HGNN_NODE model.

        Parameters
        ----------
        adj : torch.Tensor
            Adjacency matrix.
        features : torch.Tensor
            Input node features.

        Returns
        -------
        torch.Tensor
            Output node features.
        """
        node_repr = self.activation(self.input_embedding(features))
        mask = torch.ones((features.shape[0], 1)).cuda()  # [node_num, 1]
        node_repr = self.hgnn(node_repr, adj)  # [node_num, embed_size]
        _, node_centroid_sim = self.distance(node_repr, mask)  # [1, node_num, num_centroid]
        class_logit = self.output_linear(node_centroid_sim.squeeze())
        return self.log_softmax(class_logit)
