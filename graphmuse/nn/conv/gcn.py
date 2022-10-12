import torch
import torch.nn as nn

# TODO check for correctness.
class GraphConv(nn.Module):
    """GCN implementation."""
    def __init__(self, in_dim, out_dim, dropout, bias=True):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.bias = bias

        self.W = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        if self.bias:
            self.b = nn.Parameter(torch.zeros(size=(out_dim,)))
            nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        h_prime = torch.matmul(adj, Wh)
        if self.bias:
            return h_prime + self.b
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_dim) + ' -> ' \
            + str(self.out_dim) + ')'