from torch_geometric.nn import SAGEConv as SageConvLayer
import torch.nn as nn
import torch
import torch.nn.functional as F


class GGRU(nn.Module):
    """
    A GRU inspired implementation of a GCN cell.

    h(t-1) in this case is the neighbors of node t.
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GGRU, self).__init__()
        self.wr = SageConvLayer(in_features, in_features)
        self.wz = SageConvLayer(in_features, out_features)
        self.w_ni = nn.Linear(in_features*2, out_features)
        self.w_nh = nn.Linear(in_features, in_features)
        self.proj = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.w_ni.weight, gain=gain)
        nn.init.xavier_uniform_(self.w_nh.weight, gain=gain)
        nn.init.xavier_uniform_(self.proj.weight, gain=gain)

    def forward(self, x, adj):
        h = x
        r = F.sigmoid(self.wr(h, adj))
        z = F.sigmoid(self.wz(h, adj))
        h = torch.bmm(adj, self.w_nh(h)) / (adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1], -1)) + 1)
        n = self.w_ni(torch.cat([x, r*h], dim=-1))
        n = F.tanh(n)
        neigh = torch.bmm(adj, x) / (adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1], -1)) + 1)
        out = (1 - z)*n + z*self.proj(neigh)
        return out