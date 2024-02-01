import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero


# Create a GNN Encoder
class GNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, dropout=0.5):
        super(GNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, output_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(output_dim, output_dim))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.normalize = nn.LayerNorm(output_dim)

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.normalize(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return x


class MetricalGNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, metadata, dropout=0.5):
        super(MetricalGNN, self).__init__()
        self.gnn = to_hetero(GNN(input_dim, output_dim, num_layers, dropout=dropout), metadata, aggr="mean")
        self.rnn_model = nn.Sequential(
            nn.LSTM(input_dim, output_dim, batch_first=True),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.LSTM(output_dim, output_dim, batch_first=True),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x_dict, edge_index_dict, batch_note, batch_beat, subgraph_size):
        x_dict = self.gnn(x_dict, edge_index_dict)
        note = x_dict["note"]
        beat = x_dict["beat"]
        batch_beat = torch.split(batch_beat, torch.where(batch_beat[1:] != batch_beat[:-1])[0].tolist())
        # Pad the beat sequences
        h = nn.utils.rnn.pad_sequence(beat, batch_first=True)
        # Pass the beat sequences through the RNN
        h, _ = self.rnn_model(h)
        new_note = torch.empty_like(note)
        edge_bcn = edge_index_dict.pop(("beat", "connects", "note"))
        new_note[edge_bcn[0]] = h[edge_bcn[1]]
        new_note = note + new_note
        # split note where the batch_note changes
        batch_note = torch.split(new_note, torch.where(batch_note[1:] != batch_note[:-1])[0].tolist())
        # Restrict the note sequences to the subgraph size
        out = torch.cat([data[:subgraph_size] for data in batch_note], dim=0)
        # Return the output
        out = self.mlp(out)
        return out

