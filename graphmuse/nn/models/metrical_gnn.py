import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, GATConv, MessagePassing, HGTConv, JumpingKnowledge, \
    HeteroJumpingKnowledge
from graphmuse.utils.graph_utils import trim_to_layer
from torch_geometric.utils import trim_to_layer as trim_to_layer_pyg
from graphmuse.nn.conv.gat import CustomGATConv
from typing import List, Tuple, Dict, Any
from torch import Tensor, LongTensor


# Create a GNN Encoder
class HierarchicalHeteroGraphSage(torch.nn.Module):
    """
    Hierarchical Hetero GraphSage model that uses SAGEConv as the convolutional layer

    Args:
        edge_types (List[str]): List of edge types
        input_channels (int): Number of input channels
        hidden_channels (int): Number of hidden channels
        num_layers (int): Number of layers
        dropout (float, optional): Dropout rate. Defaults to 0.5.
    """
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
        """
        Forward pass of the Hierarchical Hetero GraphSage model

        Args:
            x_dict (Dict[str, Tensor]): Dictionary of node features
            edge_index_dict (Dict[str, LongTensor]): Dictionary of edge indices
            neighbor_mask_node (Dict[str, List[int]], optional): Dictionary of neighbor mask for nodes. Defaults to None.
            neighbor_mask_edge (Dict[str, List[int]], optional): Dictionary of neighbor mask for edges. Defaults to None.

        Returns:
            Dict[str, Tensor]: Output dictionary of node features
        """
        for i, conv in enumerate(self.convs[:-1]):
            if not neighbor_mask_edge is None and not neighbor_mask_node is None:
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

        if not neighbor_mask_edge is None and not neighbor_mask_node is None:
            x_dict, edge_index_dict, _ = trim_to_layer(
                layer=1,
                neighbor_mask_node=neighbor_mask_node,
                neighbor_mask_edge=neighbor_mask_edge,
                x=x_dict,
                edge_index=edge_index_dict,
            )
        x_dict = self.convs[-1](x_dict, edge_index_dict)
        return x_dict


class HierarchicalHeteroGraphConv(torch.nn.Module):
    """
    Hierarchical Hetero GraphConv model that uses CustomGATConv as the convolutional layer

    Args:
        edge_types (List[str]): List of edge types
        input_channels (int): Number of input channels
        hidden_channels (int): Number of hidden channels
        num_layers (int): Number of layers
        dropout (float, optional): Dropout rate. Defaults to 0.5.
    """
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
                }, aggr='mean'
            )
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
        """
        Forward pass of the Hierarchical Hetero GraphConv model

        Args:
            x_dict (Dict[str, Tensor]): Dictionary of node features
            edge_index_dict (Dict[str, LongTensor]): Dictionary of edge indices
            neighbor_mask_node (Dict[str, List[int]], optional): Dictionary of neighbor mask for nodes. Defaults to None.
            neighbor_mask_edge (Dict[str, List[int]], optional): Dictionary of neighbor mask for edges. Defaults to None.

        Returns:
            Dict[str, Tensor]: Output dictionary of node features
        """
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
    """
    Fast Hierarchical Hetero GraphConv model that uses SAGEConv as the convolutional layer

    Args:
        metadata (Tuple(List, List[str])): Metadata of the graph
        input_channels (int): Number of input channels
        hidden_channels (int): Number of hidden channels
        num_layers (int): Number of layers
        dropout (float, optional): Dropout rate. Defaults to 0.5.
    """
    def __init__(self, metadata, input_channels, hidden_channels, num_layers, dropout=0.5, use_jk=False):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        edge_types = metadata[1]
        if use_jk:
            self.jk = HeteroJumpingKnowledge(
                types=metadata[0], mode='lstm', channels=hidden_channels, num_layers=num_layers)
        self.use_jk = use_jk
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

    def forward(self, x_dict, edge_index_dict, neighbor_mask_node=None,
                neighbor_mask_edge=None):
        """
        Forward pass of the Fast Hierarchical Hetero GraphConv model

        Args:
            x_dict (Dict[str, Tensor]): Dictionary of node features
            edge_index_dict (Dict[str, LongTensor]): Dictionary of edge indices
            neighbor_mask_node (Dict[str, List[int]], optional): Dictionary of neighbor mask for nodes. Defaults to None.
            neighbor_mask_edge (Dict[str, List[int]], optional): Dictionary of neighbor mask for edges. Defaults to None.

        Returns:
            Dict[str, Tensor]: Output dictionary of node features
        """
        x_dict_list = {key: [] for key in x_dict}
        for i, conv in enumerate(self.convs):
            if not neighbor_mask_edge is None and not neighbor_mask_node is None:
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

            for key in x_dict.keys():
                x_dict_list[key].append(x_dict[key])

        if self.use_jk:
            batch_sizes = {k: x_dict[k].shape[0] for k in x_dict.keys()}
            x_dict_list = {k: [z[:batch_sizes[k]] for z in v] for k, v in x_dict_list.items()}
            x_dict = self.jk(x_dict_list)

        return x_dict


class GRUWrapper(nn.Module):
    """
    GRU Wrapper model that uses GRU as the recurrent layer

    Args:
        input_size (int): Number of input channels
        hidden_size (int): Number of hidden channels
        num_layers (int): Number of layers
        batch_first (bool): Whether the input is batch first
    """
    def __init__(self, input_size, hidden_size, num_layers, batch_first):
        super(GRUWrapper, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=batch_first)

    def forward(self, x):
        """
        Forward pass of the GRU Wrapper model

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Output tensor
        """
        output, _ = self.gru(x)
        return output


class MetricalGNN(nn.Module):
    """
    MetricalGNN model that uses hierarchical hetero graph convolutional layers

    Args:
        input_channels (int): Number of input channels
        hidden_channels (int): Number of hidden channels
        output_channels (int): Number of output channels
        num_layers (int): Number of layers
        metadata (Tuple[Dict[str, int], List[Tuple[str, str]]]): Metadata of the graph
        dropout (float, optional): Dropout rate. Defaults to 0.5.
        fast (bool, optional): Whether to use FastHierarchicalHeteroGraphConv. Defaults to False.
        remove_metrical_features: Whether to use features from metrical nodes or directly learn from neighbor features.
    """
    def __init__(self, input_channels, hidden_channels, output_channels, num_layers, metadata, dropout=0.5,
                 fast=False, remove_metrical_features=False, use_jk=False):
        super(MetricalGNN, self).__init__()
        self.num_layers = num_layers

        if fast:
            self.gnn = FastHierarchicalHeteroGraphConv(metadata, input_channels, hidden_channels, num_layers, dropout=dropout, use_jk=use_jk)
            self.fhs = True
        else:
            self.gnn = HierarchicalHeteroGraphSage(metadata[1], input_channels, hidden_channels, num_layers, dropout=dropout)
            self.fhs = False
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, output_channels)
        )
        self.remove_metrical_features = remove_metrical_features

    def forward(self, x_dict, edge_index_dict, neighbor_mask_node=None, neighbor_mask_edge=None):
        """
        Forward pass of the MetricalGNN model

        Args:
            x_dict (Dict[str, Tensor]): Dictionary of node features
            edge_index_dict (Dict[str, LongTensor]): Dictionary of edge indices
            neighbor_mask_node (Dict[str, List[int]], optional): Dictionary of neighbor mask for nodes. Defaults to None.
            neighbor_mask_edge (Dict[str, List[int]], optional): Dictionary of neighbor mask for edges. Defaults to None.

        Returns:
            Tensor: Output tensor
        """
        x_dict = self.gnn(x_dict, edge_index_dict, neighbor_mask_node, neighbor_mask_edge)
        note = x_dict["note"]
        # Return the output
        out = self.mlp(note)
        return out


class HybridGNN(torch.nn.Module):
    """
    Hybrid GNN model that uses SAGEConv as the convolutional layer

    Args:
        edge_types (List[str]): List of edge types
        input_channels (int): Number of input channels
        hidden_channels (int): Number of hidden channels
        num_layers (int): Number of layers
        dropout (float, optional): Dropout rate. Defaults to 0.5.
    """
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
        """
        Forward pass of the Hybrid GNN model

        Args:
            x (Tensor): Input tensor
            batch (Tensor): Batch tensor

        Returns:
            Tensor: Output tensor
        """
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
        """
        Forward pass of the Hybrid GNN model

        Args:
            x_dict (Dict[str, Tensor]): Dictionary of node features
            edge_index_dict (Dict[str, LongTensor]): Dictionary of edge indices
            batch_dict (Dict[str, Tensor]): Dictionary of batch tensors
            batch_size (int): Batch size
            neighbor_mask_node (Dict[str, List[int]], optional): Dictionary of neighbor mask for nodes. Defaults to None.
            neighbor_mask_edge (Dict[str, List[int]], optional): Dictionary of neighbor mask for edges. Defaults to None.
            return_edge_index (bool, optional): Whether to return edge index. Defaults to False.

        Returns:
            Tensor: Output tensor
        """
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
        """
        Forward pass of the Hybrid GNN model

        Args:
            x (Tensor): Input tensor
            batch (Tensor): Batch tensor

        Returns:
            Tensor: Output tensor
        """
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
        """
        Forward pass of the Hybrid GNN model

        Args:
            x_dict (Dict[str, Tensor]): Dictionary of node features
            edge_index_dict (Dict[str, LongTensor]): Dictionary of edge indices
            batch_dict (Dict[str, Tensor]): Dictionary of batch tensors
            batch_size (int, optional): Batch size. Defaults to None.
            neighbor_mask_node (Dict[str, List[int]], optional): Dictionary of neighbor mask for nodes. Defaults to None.
            neighbor_mask_edge (Dict[str, List[int]], optional): Dictionary of neighbor mask for edges. Defaults to None.
            return_edge_index (bool, optional): Whether to return edge index. Defaults to False.

        Returns:
            Tensor: Output tensor
        """
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
