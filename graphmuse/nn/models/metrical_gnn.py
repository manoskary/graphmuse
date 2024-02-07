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
        # Return the output
        out = self.mlp(note)
        return out

