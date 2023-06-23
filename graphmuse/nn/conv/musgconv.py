import torch.nn as nn
from torch_scatter import scatter

class RelEdgeConv(nn.Module):
    def __init__(self, in_node_features, out_features, bias=True, in_edge_features=None):
        super(RelEdgeConv, self).__init__()
        self.neigh_linear = nn.Linear(in_node_features, in_node_features, bias=bias)
        self.edge_linear = nn.Linear((in_node_features*2 if in_edge_features is None else in_node_features+in_edge_features), in_node_features, bias=bias)
        self.linear = nn.Linear(in_node_features*2, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        nn.init.xavier_uniform_(self.neigh_linear.weight, gain=gain)
        nn.init.xavier_uniform_(self.edge_linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.)
        if self.neigh_linear.bias is not None:
            nn.init.constant_(self.neigh_linear.bias, 0.)
        if self.edge_linear.bias is not None:
            nn.init.constant_(self.edge_linear.bias, 0.)

    def forward(self, features, edge_index, edge_features=None):
        h = self.neigh_linear(features)
        if edge_features is None:
            edge_features = torch.abs(h[edge_index[0]] - h[edge_index[1]])
        new_h = self.edge_linear(torch.cat((h[edge_index[1]], edge_features), dim=-1))
        s = scatter(new_h, edge_index[0], 0, out=h.clone(), reduce='mean')
        z = self.linear(torch.cat([features, s], dim=-1))
        return z