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


class Transformer(nn.Module):
    """Transformer model for uneven sequences that works with nested tensors as input."""
    def __init__(self, input_dim, output_dim, num_layers, dropout=0.5):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList()


class GRUWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first):
        super(GRUWrapper, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=batch_first)

    def forward(self, x):
        output, _ = self.gru(x)
        return output


class MetricalGNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, metadata, dropout=0.5):
        super(MetricalGNN, self).__init__()
        self.gnn = to_hetero(GNN(input_dim, output_dim, num_layers, dropout=dropout), metadata, aggr="mean")
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3, norm=nn.LayerNorm(output_dim),
                                                         enable_nested_tensor=False)
        # self.gru_model = nn.Sequential(
        #     GRUWrapper(input_size=output_dim, hidden_size=output_dim, num_layers=1, batch_first=True),
        #     nn.ReLU(),
        #     nn.LayerNorm(output_dim),
        #     nn.Dropout(dropout),
        #     GRUWrapper(input_size=output_dim, hidden_size=output_dim, num_layers=1, batch_first=True),
        # )
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
        h_beat = x_dict["beat"]
        splits = [0] + torch.where(batch_beat[1:] != batch_beat[:-1])[0].tolist() + [len(batch_beat)]
        # take diff of splits to get the length of each beat sequence with next element
        splits = [s - e for s, e in zip(splits[1:], splits[:-1])]
        h_beat = torch.split(h_beat, splits)
        # pad the beat sequences
        h = nn.utils.rnn.pad_sequence(h_beat, batch_first=True)
        # Pass the beat sequences through the transformer or gru layer
        h = self.transformer_encoder(h)
        # h = self.gru_model(h)
        # Unpad the beat sequences
        h = nn.utils.rnn.unpad_sequence(h, lengths=torch.as_tensor(splits), batch_first=True)
        h = torch.cat(h)
        new_note = torch.empty_like(note)
        edge_bcn = edge_index_dict.pop(("beat", "connects", "note"))
        new_note[edge_bcn[1]] = h[edge_bcn[0]]
        new_note = note + new_note
        # split note where the batch_note changes
        note_splits = [0] + torch.where(batch_note[1:] != batch_note[:-1])[0].tolist() + [len(batch_note)]
        note_splits = [s - e for s, e in zip(note_splits[1:], note_splits[:-1])]
        batch_note = torch.split(new_note, note_splits)
        # Restrict the note sequences to the subgraph size
        out = torch.cat([data[:subgraph_size] for data in batch_note], dim=0)
        # Return the output
        out = self.mlp(out)
        return out

