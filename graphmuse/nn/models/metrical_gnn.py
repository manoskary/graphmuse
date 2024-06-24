import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, GATConv, MessagePassing, HGTConv, JumpingKnowledge
from graphmuse.utils.graph_utils import trim_to_layer
from torch_geometric.utils import trim_to_layer as trim_to_layer_pyg
from graphmuse.nn.conv.gat import CustomGATConv
from typing import List, Tuple, Dict, Any
from torch import Tensor, LongTensor


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


class HybridGNN(torch.nn.Module):
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

        self.rnn = nn.GRU(
            input_size=input_channels, hidden_size=hidden_channels // 2, num_layers=2, batch_first=True, bidirectional=True,
            dropout=dropout)
        self.rnn_norm = nn.LayerNorm(hidden_channels)
        self.rnn_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.cat_proj = nn.Linear(hidden_channels * 2, hidden_channels)

    def hybrid_forward(self, x, batch):
        # NOTE optimize sampling to order sequences by length
        lengths = torch.bincount(batch)
        x = x.split(lengths.tolist())
        x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.0)
        x, _ = self.rnn(x)
        x = self.rnn_norm(x)
        x = self.rnn_mlp(x)
        x = nn.utils.rnn.unpad_sequence(x, batch_first=True, lengths=lengths)
        x = torch.cat(x, dim=0)
        return x

    def forward(self, x_dict, edge_index_dict, batch_dict, batch_size, neighbor_mask_node,
                neighbor_mask_edge, return_edge_index=False):
        if batch_dict is None:
            batch_dict = {key: torch.zeros(x.size(0), dtype=torch.long, device=x.device) for key, x in x_dict.items()}

        x_note_target = x_dict["note"][:batch_size]
        batch_note = batch_dict["note"][:batch_size]
        x = self.hybrid_forward(x_note_target, batch_note)

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
        x_gnn = x_dict["note"][:batch_size]
        x = self.cat_proj(torch.cat([x, x_gnn], dim=-1))
        if return_edge_index:
            # Trim Edge Index to only notes and remove edge_indices > batch_size
            edge_index_dict = {key: edge_index_dict[key][:, edge_index_dict[key][0] < batch_size] for key in edge_index_dict if (key[0] == "note" and key[-1] == "note")}
            return x, edge_index_dict
        return x


class HybridHGT(torch.nn.Module):
    """
    Hybrid GNN model that uses HGTConv as the convolutional layer

    The Hybrid GNN model mixes the GNN model with an RNN model to capture the temporal dependencies in the data.
    It was introduced in the paper "GraphMuse: A Library for Symbolic Music Graph Processing"

    Args:
        metadata (Tuple[Dict[str, int], List[Tuple[str, str]]]): Metadata of the graph
        input_channels (int): Number of input channels
        hidden_channels (int): Number of hidden channels
        num_layers (int): Number of layers
        heads (int, optional): Number of attention heads. Defaults to 4.
        dropout (float, optional): Dropout rate. Defaults to 0.5.
        jk (bool, optional): Whether to use Jumping Knowledge. Defaults to False.
    """
    def __init__(
            self,
            metadata: Tuple[List[str], List[Tuple[str, str, str]]],
            input_channels: int,
            hidden_channels: int,
            num_layers: int,
            heads: int = 4,
            dropout: float = 0.5,
            jk = False

    ):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.convs.append(HGTConv(input_channels, hidden_channels, metadata, heads=heads))
        for _ in range(num_layers - 1):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, heads=heads)
            self.convs.append(conv)
            self.layer_norms.append(nn.LayerNorm(hidden_channels))

        self.rnn = nn.GRU(
            input_size=input_channels, hidden_size=hidden_channels // 2, num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout)
        self.rnn_norm = nn.LayerNorm(hidden_channels)
        self.rnn_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.cat_proj = nn.Linear(hidden_channels * 2, hidden_channels)
        if jk:
            self.jk = JumpingKnowledge(mode='lstm', channels=hidden_channels, num_layers=num_layers)

    def hybrid_forward(self, x, batch):
        # NOTE optimize sampling to order sequences by length
        lengths = torch.bincount(batch)
        x = x.split(lengths.tolist())
        x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.0)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        x, _ = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=0.0)
        x = self.rnn_norm(x)
        x = self.rnn_mlp(x)
        x = nn.utils.rnn.unpad_sequence(x, batch_first=True, lengths=lengths)
        x = torch.cat(x, dim=0)
        return x

    def forward(self, x_dict, edge_index_dict, batch_dict, batch_size=None, neighbor_mask_node=None,
                neighbor_mask_edge=None, return_edge_index=False):
        xs = []
        if batch_dict is None:
            batch_dict = {key: torch.zeros(x.size(0), dtype=torch.long, device=x.device) for key, x in x_dict.items()}
        if neighbor_mask_node is None:
            neighbor_mask_node = {key: [x_dict[key].size(0)] for key in x_dict}

        batch_size = neighbor_mask_node["note"][0] if batch_size is None else batch_size
        x_note_target = x_dict["note"][:batch_size]
        batch_note = batch_dict["note"][:batch_size]
        x = self.hybrid_forward(x_note_target, batch_note)
        for i, conv in enumerate(self.convs):
            if neighbor_mask_edge is not None:
                x_dict, edge_index_dict, _ = trim_to_layer_pyg(
                    layer=i,
                    num_sampled_edges_per_hop=neighbor_mask_edge,
                    num_sampled_nodes_per_hop=neighbor_mask_node,
                    x=x_dict,
                    edge_index=edge_index_dict,
                )
            x_dict = conv(x_dict, edge_index_dict)
            if i != len(self.convs) - 1:
                x_dict = {key: self.layer_norms[i](x) for key, x in x_dict.items()}
                x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

            if hasattr(self, 'jk'):
                xs.append(x_dict["note"][:batch_size])

        x_gnn = self.jk(xs) if hasattr(self, 'jk') else x_dict["note"][:batch_size]
        x = self.cat_proj(torch.cat([x, x_gnn], dim=-1))
        if return_edge_index:
            # Trim Edge Index to only notes and remove edge_indices > batch_size
            edge_index_dict = {key: edge_index_dict[key][:, edge_index_dict[key][0] < batch_size] for key in edge_index_dict if (key[0] == "note" and key[-1] == "note")}
            return x, edge_index_dict
        return x
