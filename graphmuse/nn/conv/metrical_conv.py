from .sage import SageConvLayer
import torch.nn as nn
import torch
from torch_scatter import scatter_add
from torch_geometric.nn import SAGEConv


class MetricalConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None, dropout=0.2, bias=True):
        super().__init__()
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.activation = nn.Identity() if activation is None else activation
        self.dropout = nn.Dropout(dropout)
        self.normalize = nn.BatchNorm1d(out_dim)
        self.neigh = nn.Linear(in_dim, in_dim, bias=bias)
        self.conv_out = nn.Linear(3 * in_dim, out_dim, bias=bias)
        self.seq = SAGEConv(in_dim, in_dim, aggr='add')

    def reset_parameters(self):
        self.neigh.reset_parameters()
        self.conv_out.reset_parameters()
        self.seq.reset_parameters()

    def forward(self, x_metrical, x, edge_index, batch):
        if batch is None or torch.all(batch == batch[0]):
            seq_index = torch.vstack((torch.arange(0, x_metrical.size(0) - 1), torch.arange(1, x_metrical.size(0)))).long()
        else:
            seq_index = []
            lengths = torch.unique(batch, return_counts=True)[1]
            # add zero to the beginning of lengths
            lengths = torch.cat((torch.zeros(1, dtype=torch.long), lengths))
            lengths_cummulative = torch.cumsum(lengths, dim=0)
            for i in range(len(lengths) - 1):
                if lengths[i] == 1:
                    continue
                seq_index.append(torch.vstack((torch.arange(lengths_cummulative[i], lengths_cummulative[i + 1] - 1), torch.arange(lengths_cummulative[i] + 1, lengths_cummulative[i + 1]))))
            seq_index = torch.cat(seq_index, dim=1)
            seq_index = torch.cat((seq_index, torch.vstack((seq_index[1], seq_index[0]))), dim=1).long()

        h_neigh = self.neigh(x)
        h_scatter = scatter_add(h_neigh[edge_index[0]], edge_index[1], dim=0, dim_size=x_metrical.size(0))
        h_seq = self.seq(x_metrical, seq_index.to(x_metrical.device))
        h = torch.cat([h_scatter, x_metrical, h_seq], dim=1)
        h = self.conv_out(h)
        h = self.activation(h)
        h = self.normalize(h)
        h = self.dropout(h)
        out = scatter_add(h[edge_index[1]], edge_index[0], dim=0, dim_size=x.size(0))
        return out