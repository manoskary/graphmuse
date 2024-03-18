import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, GATConv, MessagePassing
from graphmuse.utils.graph_utils import trim_to_layer
from torch_geometric.utils import trim_to_layer as trim_to_layer_pyg
from graphmuse.nn.conv.gat import CustomGATConv


# Create a GNN Encoder
class HierarchicalHeteroGraphSage(torch.nn.Module):
    def __init__(self, edge_types, input_channels, hidden_channels, num_layers, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.convs.append(
            HeteroConv(
                {
                    edge_type: SAGEConv(input_channels, hidden_channels, normalize=True, project=True)
                    for edge_type in edge_types
                }, aggr='mean')
        )
        for _ in range(num_layers-1):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv(hidden_channels, hidden_channels)
                    for edge_type in edge_types
                }, aggr='mean')
            self.convs.append(conv)
            self.layer_norms.append(nn.LayerNorm(hidden_channels))

    def forward(self, x_dict, edge_index_dict, neighbor_mask_node,
                neighbor_mask_edge):

        for i, conv in enumerate(self.convs[:-1]):
            x_dict, edge_index_dict, _ = trim_to_layer(
                layer=self.num_layers - i,
                neighbor_mask_node=neighbor_mask_node,
                neighbor_mask_edge=neighbor_mask_edge,
                x=x_dict,
                edge_index=edge_index_dict,
            )

            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict = {key: self.layer_norms[i](x) for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        # Last layer
        x_dict, edge_index_dict, _ = trim_to_layer(
            layer=1,
            neighbor_mask_node=neighbor_mask_node,
            neighbor_mask_edge=neighbor_mask_edge,
            x=x_dict,
            edge_index=edge_index_dict,
        )

        return x_dict


class HierarchicalHeteroGraphConv(torch.nn.Module):
    def __init__(self, edge_types, input_channels, hidden_channels, num_layers, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.convs.append(
            HeteroConv(
                {
                    edge_type: CustomGATConv(input_channels, hidden_channels, heads=4, add_self_loops=False)
                    for edge_type in edge_types
                }, aggr='mean')
        )
        for _ in range(num_layers-1):
            conv = HeteroConv(
                {
                    edge_type: CustomGATConv(hidden_channels, hidden_channels, heads=4, add_self_loops=False)
                    for edge_type in edge_types
                }, aggr='mean')
            self.convs.append(conv)
            self.layer_norms.append(nn.LayerNorm(hidden_channels))

    def forward(self, x_dict, edge_index_dict, neighbor_mask_node,
                neighbor_mask_edge):

        for i, conv in enumerate(self.convs[:-1]):
            x_dict, edge_index_dict, _ = trim_to_layer(
                layer=self.num_layers - i,
                neighbor_mask_node=neighbor_mask_node,
                neighbor_mask_edge=neighbor_mask_edge,
                x=x_dict,
                edge_index=edge_index_dict,
            )

            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict = {key: self.layer_norms[i](x) for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        # Last layer
        x_dict, edge_index_dict, _ = trim_to_layer(
            layer=1,
            neighbor_mask_node=neighbor_mask_node,
            neighbor_mask_edge=neighbor_mask_edge,
            x=x_dict,
            edge_index=edge_index_dict,
        )
        x_dict = self.convs[-1](x_dict, edge_index_dict)
        return x_dict


class FastHierarchicalHeteroGraphConv(torch.nn.Module):
    def __init__(self, edge_types, input_channels, hidden_channels, num_layers, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.convs.append(
            HeteroConv(
                {
                    edge_type: SAGEConv(input_channels, hidden_channels, normalize=True, project=True)
                    for edge_type in edge_types
                }, aggr='mean')
        )
        for _ in range(num_layers - 1):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv(hidden_channels, hidden_channels)
                    for edge_type in edge_types
                }, aggr='mean')
            self.convs.append(conv)
            self.layer_norms.append(nn.LayerNorm(hidden_channels))

    def forward(self, x_dict, edge_index_dict, neighbor_mask_node,
                neighbor_mask_edge):

        for i, conv in enumerate(self.convs):
            x_dict, edge_index_dict, _ = trim_to_layer_pyg(
                layer=i,
                num_sampled_edges_per_hop=neighbor_mask_edge,
                num_sampled_nodes_per_hop=neighbor_mask_node,
                x=x_dict,
                edge_index=edge_index_dict,
            )
            x_dict = conv(x_dict, edge_index_dict)
            if i != len(self.convs) - 1:
                x_dict = {key: x.relu() for key, x in x_dict.items()}
                x_dict = {key: self.layer_norms[i](x) for key, x in x_dict.items()}
                x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        return x_dict


class GRUWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first):
        super(GRUWrapper, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=batch_first)

    def forward(self, x):
        output, _ = self.gru(x)
        return output


class MetricalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, metadata, dropout=0.5, fast=False):
        super(MetricalGNN, self).__init__()
        if fast:
            self.gnn = FastHierarchicalHeteroGraphConv(metadata[1], input_dim, hidden_dim, num_layers)
        else:
            self.gnn = HierarchicalHeteroGraphSage(metadata[1], input_dim, hidden_dim, num_layers)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3, norm=nn.LayerNorm(hidden_dim),
        #                                                  enable_nested_tensor=False)
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
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_dict, edge_index_dict, neighbor_mask_node, neighbor_mask_edge):
        x_dict = self.gnn(x_dict, edge_index_dict, neighbor_mask_node, neighbor_mask_edge)
        note = x_dict["note"]
        # Return the output
        out = self.mlp(note)
        return out
