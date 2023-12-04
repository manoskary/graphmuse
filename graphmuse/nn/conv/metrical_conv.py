from .sage import SageConvLayer
import torch.nn as nn
import torch
from torch_geometric.nn import MessagePassing


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
        self.seq = SageConvLayer(in_dim, in_dim, bias=bias)
        self.msp_in = MessagePassing(aggr='add')
        self.msp_out = MessagePassing(aggr='add')

    def reset_parameters(self):
        self.neigh.reset_parameters()
        self.conv_out.reset_parameters()
        self.seq.reset_parameters()

    def forward(self, x_metrical, x, edge_index, lengths=None):
        if lengths is None:
            seq_index = torch.vstack((torch.arange(0, x_metrical.shape[0] - 1), torch.arange(1, x_metrical.shape[0])))
            # add inverse
            seq_index = torch.cat((seq_index, torch.vstack((seq_index[1], seq_index[0]))), dim=1).long()
        else:
            seq_index = []
            for i in range(len(lengths) - 1):
                seq_index.append(torch.vstack((torch.arange(lengths[i], lengths[i + 1] - 1), torch.arange(lengths[i] + 1, lengths[i + 1]))))
            seq_index = torch.cat(seq_index, dim=1)
            seq_index = torch.cat((seq_index, torch.vstack((seq_index[1], seq_index[0]))), dim=1).long()
        h_neigh = self.neigh(x)
        h_scatter = self.msp_in(h_neigh, edge_index, size=(x_metrical.size(0), self.input_dim))
        # h_scatter = scatter(h_neigh[edge_index[0]], edge_index[1], dim=0, out=torch.zeros(x_metrical.size(0), self.input_dim, dtype=x.dtype).to(x.device))
        h_seq = self.seq(x_metrical, seq_index.to(x_metrical.device))
        h = torch.cat([h_scatter, x_metrical, h_seq], dim=1)
        h = self.conv_out(h)
        h = self.activation(h)
        h = self.normalize(h)
        h = self.dropout(h)
        out = self.msp_out(h, edge_index.flip(1), size=(x.size(0), self.output_dim))
        # out = scatter(h[edge_index[1]], edge_index[0], dim=0, out=torch.zeros(x.size(0), self.output_dim, dtype=h.dtype).to(x.device))
        return out, h