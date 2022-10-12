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
    def __init__(self, in_feats, out_feats):
        super(CentroidDistance, self).__init__()
        # centroid embedding
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.centroid_embedding = nn.Embedding(in_feats, out_feats,
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
        node_repr = node_repr.unsqueeze(1).expand(-1, self.in_feats, -1).contiguous().view(-1, self.out_feats)

        # broadcast and reshape centroid embeddings to [node_num * num_centroid, embed_size]
        centroid_repr = self.centroid_embedding(torch.arange(self.in_feats).cuda())
        centroid_repr = centroid_repr.unsqueeze(0).expand(node_num, -1,-1).contiguous().view(-1, self.out_feats)
        # get distance
        node_centroid_dist = self.manifold.distance(node_repr, centroid_repr)
        node_centroid_dist = node_centroid_dist.view(1, node_num, self.out_feats) * mask
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
    def __init__(self, in_feats, out_feats, bias=True):
        super(HyperbolicSageConv, self).__init__()
        self.bias = bias
        self.embed = torch.zeros((in_feats, in_feats), requires_grad=True)
        self.layer = torch.zeros((2*in_feats, out_feats), requires_grad=True)
        self.reset_parameters()
        self.embed = nn.Parameter(self.embed)
        self.layer = nn.Parameter(self.layer)
        if self.bias:
            self.embed_bias = torch.zeros((in_feats, 1), requires_grad=True)
            self.layer_bias = torch.zeros((out_feats, 1), requires_grad=True)

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
    def __init__(self, in_feats, n_hidden, out_feats, n_layers, activation=F.relu, dropout=0.5):
        super(HGNN, self).__init__()
        self.init_layer = exp_map_zero
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(HyperbolicSageConv(in_feats, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(HyperbolicSageConv(n_hidden, n_hidden))
        self.layers.append(HyperbolicSageConv(n_hidden, out_feats))

    def forward(self, x, adj):
        # 0-projection to Poincare Hyperbolic space.
        h = self.init_layer(x)
        for i, layer in enumerate(self.layers):
            h = layer(x, adj)
            if i != len(layer)-1:
                h = exp_map_zero(self.activation(log_map_zero(h)))
                h = self.dropout(h)
        return h

class HGNN_NODE(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers, activation=F.relu, dropout=0.5):
        self.input_embedding  = nn.Linear(in_feats, n_hidden)
        self.hgnn = HGNN(n_hidden, n_hidden, n_hidden, n_layers, activation, dropout)
        self.distance = CentroidDistance(n_hidden, n_hidden)
        self.output_linear = nn.Linear(n_hidden, n_hidden)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.input_embedding.weight, gain=gain)
        nn.init.xavier_uniform_(self.output_linear.weight, gain=gain)

    def forward(self, adj, features):
        """
        Args:
            adj: the neighbor ids of each node [1, node_num, max_neighbor]
            weight: the weight of each neighbor [1, node_num, max_neighbor]
            features: [1, node_num, input_dim]
        """

        node_repr = self.activation(self.input_embedding(features))
        mask = torch.ones((features.shape[0], 1)).cuda()  # [node_num, 1]
        node_repr = self.hgnn(node_repr, adj)  # [node_num, embed_size]
        _, node_centroid_sim = self.distance(node_repr, mask)  # [1, node_num, num_centroid]
        class_logit = self.output_linear(node_centroid_sim.squeeze())
        return self.log_softmax(class_logit)