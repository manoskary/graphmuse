import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
from graphmuse.utils.graph_utils import trim_to_layer


# Create a GNN Encoder
class HierarchicalHeteroGraphSage(torch.nn.Module):
    def __init__(self, edge_types, hidden_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((-1, -1), hidden_channels)
                    for edge_type in edge_types
                }, aggr='sum')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict, neighbor_mask_node,
                neighbor_mask_edge):

        for i, conv in enumerate(self.convs):
            x_dict, edge_index_dict, _ = trim_to_layer(
                layer=self.num_layers - i,
                neighbor_mask_node=neighbor_mask_node,
                neighbor_mask_edge=neighbor_mask_edge,
                x=x_dict,
                edge_index=edge_index_dict,
                padding_value=0.0
            )

            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict


class GRUWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first):
        super(GRUWrapper, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=batch_first)

    def forward(self, x):
        output, _ = self.gru(x)
        return output


class MetricalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, metadata, dropout=0.5):
        super(MetricalGNN, self).__init__()
        self.gnn = HierarchicalHeteroGraphSage(metadata[1], hidden_dim, num_layers)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3, norm=nn.LayerNorm(hidden_dim),
                                                         enable_nested_tensor=False)
        # self.gru_model = nn.Sequential(
        #     GRUWrapper(input_size=output_dim, hidden_size=output_dim, num_layers=1, batch_first=True),
        #     nn.ReLU(),
        #     nn.LayerNorm(output_dim),
        #     nn.Dropout(dropout),
        #     GRUWrapper(input_size=output_dim, hidden_size=output_dim, num_layers=1, batch_first=True),
        # )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_dict, edge_index_dict, batch_beat, neighbor_mask_node, neighbor_mask_edge):
        x_dict = self.gnn(x_dict, edge_index_dict, neighbor_mask_node, neighbor_mask_edge)
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
        # Return the output
        out = self.mlp(new_note)
        return out

