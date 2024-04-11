import torch.nn as nn
import torch
from torch_geometric.nn import MessagePassing, MessageNorm


class MusGConv(MessagePassing):
    def __init__(self, in_channels, out_channels, in_edge_channels=0, bias=True, return_edge_emb=False, norm_msg=False, **kwargs):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.return_edge_emb = return_edge_emb
        self.aggregation = kwargs.get("aggregation", "cat")
        self.in_edge_channels = in_edge_channels if in_edge_channels > 0 else in_channels
        self.lin = nn.Linear(in_channels, out_channels)
        self.msg_norm = MessageNorm() if norm_msg else nn.Identity()
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.in_edge_channels, out_channels),
            nn.ReLU(),
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, out_channels),
        )
        self.proj = nn.Linear(3 * out_channels, out_channels) if self.aggregation == "cat" else nn.Linear(2 * out_channels, out_channels)
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
        if self.aggregation == "cat":
            msg = torch.cat((x_j, edge_attr), dim=-1)
        elif self.aggregation == "add":
            msg = x_j + edge_attr
        elif self.aggregation == "mul":
            msg = x_j * edge_attr
        else:
            raise ValueError("Aggregation type not supported")

        msg = self.msg_norm(x_j, msg)
        return msg

