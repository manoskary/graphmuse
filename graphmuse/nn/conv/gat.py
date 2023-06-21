import torch.nn as nn
import torch
from torch_scatter import scatter


# TODO check for correctness
class GraphAttentionLayer(nn.Module):
    """GAT implementation."""
    def __init__(self, in_dim, out_dim, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=1)
        attention = nn.functional.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return nn.functional.elu(h_prime)
        else:
            return h_prime


class GATConvLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=3, bias=True, dropout=0.3, negative_slope=0.2, in_edge_features=None):
        super(GATConvLayer, self).__init__()
        self.num_heads = num_heads
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.el = nn.Linear(in_features, in_features * num_heads, bias=bias)
        self.er = nn.Linear(in_features, in_features * num_heads, bias=bias)
        self.attnl = nn.Parameter(torch.FloatTensor(1, num_heads, in_features))
        self.attnr = nn.Parameter(torch.FloatTensor(1, num_heads, in_features))
        if in_edge_features is not None:
            self.attne = nn.Parameter(torch.FloatTensor(1, num_heads, in_features))
            self.fc_fij = nn.Linear(in_edge_features, in_features * num_heads, bias=bias)
        self.in_edge_feats = in_edge_features
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.attndrop = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        nn.init.xavier_normal_(self.el.weight, gain=gain)
        nn.init.xavier_normal_(self.er.weight, gain=gain)
        nn.init.xavier_normal_(self.attnl, gain=gain)
        nn.init.xavier_normal_(self.attnr, gain=gain)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.)
        if self.el.bias is not None:
            nn.init.constant_(self.el.bias, 0.)
        if self.er.bias is not None:
            nn.init.constant_(self.er.bias, 0.)
        if self.in_edge_feats is not None:
            nn.init.xavier_normal_(self.fc_fij.weight, gain=gain)
            if self.fc_fij.bias is not None:
                nn.init.constant_(self.fc_fij.bias, 0.)

    def forward(self, features, edge_index, edge_features=None):
        prefix_shape = features.shape[:-1]
        fc_src = self.el(features).view(*prefix_shape, self.num_heads, self.in_features)
        fc_dst = self.er(features).view(*prefix_shape, self.num_heads, self.in_features)
        el = (fc_src[edge_index[0]] * self.attnl).sum(dim=-1).unsqueeze(-1)
        er = (fc_dst[edge_index[1]] * self.attnr).sum(dim=-1).unsqueeze(-1)
        if edge_features is not None and self.in_edge_feats is not None:
            edge_shape = edge_features.shape[:-1]
            fc_eij = self.fc_fij(edge_features).view(*edge_shape, self.num_heads, self.in_features)
            ee = (fc_eij * self.attne).sum(dim=-1).unsqueeze(-1)
            e = self.leaky_relu(el + er + ee)
        else:
            e = self.leaky_relu(el + er)
        # Not Quite the same as the Softmax in the paper.
        a = self.softmax(self.attndrop(e)).mean(dim=1)
        h = self.linear(features)
        out = scatter(a * h[edge_index[1]], edge_index[0], 0, out=h.clone(), reduce='add')
        return out