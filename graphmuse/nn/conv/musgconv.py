import torch.nn as nn
import torch
from torch_geometric.nn import MessagePassing


class MusGConv(MessagePassing):
    def __init__(self, in_channels, out_channels, in_edge_channels=0, bias=True, return_edge_emb=False, **kwargs):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.return_edge_emb = return_edge_emb
        self.in_edge_channels = in_edge_channels if in_edge_channels > 0 else in_channels
        self.lin = nn.Linear(in_channels, out_channels)
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.in_edge_channels, out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Linear(out_channels, out_channels),
        )
        self.proj = nn.Linear(3 * out_channels, out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.lin.weight, gain=gain)
        nn.init.xavier_uniform_(self.proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.edge_mlp[0].weight, gain=gain)
        nn.init.xavier_uniform_(self.edge_mlp[3].weight, gain=gain)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is None:
            edge_attr = torch.abs(x[edge_index[0]] - x[edge_index[1]])
        x = self.lin(x)
        edge_attr = self.edge_mlp(edge_attr)
        h = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        h = self.proj(torch.cat((x, h), dim=-1))
        if self.bias is not None:
            h = h + self.bias
        if self.return_edge_emb:
            return h, edge_attr
        return h

    def message(self, x_j, edge_attr):
        return torch.cat((x_j, edge_attr), dim=-1)


# class RelEdgeConv(nn.Module):
#     def __init__(self, in_node_features, out_features, bias=True, in_edge_features=None):
#         super(RelEdgeConv, self).__init__()
#         self.neigh_linear = nn.Linear(in_node_features, in_node_features, bias=bias)
#         self.edge_linear = nn.Linear((in_node_features*2 if in_edge_features is None else in_node_features+in_edge_features), in_node_features, bias=bias)
#         self.linear = nn.Linear(in_node_features*2, out_features, bias=bias)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         gain = nn.init.calculate_gain('relu')
#         nn.init.xavier_uniform_(self.linear.weight, gain=gain)
#         nn.init.xavier_uniform_(self.neigh_linear.weight, gain=gain)
#         nn.init.xavier_uniform_(self.edge_linear.weight, gain=gain)
#         if self.linear.bias is not None:
#             nn.init.constant_(self.linear.bias, 0.)
#         if self.neigh_linear.bias is not None:
#             nn.init.constant_(self.neigh_linear.bias, 0.)
#         if self.edge_linear.bias is not None:
#             nn.init.constant_(self.edge_linear.bias, 0.)
#
#     def forward(self, features, edge_index, edge_features=None):
#         h = self.neigh_linear(features)
#         if edge_features is None:
#             edge_features = torch.abs(h[edge_index[0]] - h[edge_index[1]])
#         new_h = self.edge_linear(torch.cat((h[edge_index[1]], edge_features), dim=-1))
#         s = scatter(new_h, edge_index[0], 0, out=h.clone(), reduce='mean')
#         z = self.linear(torch.cat([features, s], dim=-1))
#         return z