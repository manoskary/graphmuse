from scipy import sparse as sp
from scipy.sparse.linalg import eigs
import torch
from typing import Dict, Optional
import numpy as np
import graphmuse.samplers as sam
from .graph import create_score_graph
from torch_geometric.typing import (
    # MaybeHeteroAdjTensor,
    # MaybeHeteroEdgeTensor,
    MaybeHeteroNodeTensor,
    NodeType,
    SparseTensor,
)


def degree(edge_index, num_nodes):
    """Computes the (unweighted) degree of a given one-dimensional index
    tensor.

    Parameters
    ----------
    index (LongTensor)
        Index tensor.

    num_nodes : (int)
        The number of nodes of the graph.
    """
    out = torch.zeros((num_nodes,), dtype=torch.long)
    one = torch.ones((edge_index[1].size(0), ), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, edge_index[1], one)


def positional_encoding(edge_index, num_nodes, pos_enc_dim: int) -> torch.Tensor:
    """
    Graph positional encoding v/ Laplacian eigenvectors

    Parameters
    ----------
    edge_index (LongTensor)
        edge indices tensor.
    num_nodes : (int)
        The number of nodes of the graph.
    pos_enc_dim : int
        The Positional Encoding Dimension to be added to Nodes of the graph

    Returns
    -------
    pos_enc : torch.Tensor
        A tensor of shape (N, pos_enc_dim) where N is the number of nodes in the graph
    """
    # Laplacian
    A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), (num_nodes, num_nodes)).to_dense()
    in_degree = A.sum(axis=1).numpy()
    N = sp.diags(in_degree.clip(1) ** -0.5, dtype=float)
    L = sp.eye(num_nodes) - N * A * N

    # Eigenvectors with scipy
    EigVal, EigVec = eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2)
    # increasing order
    EigVec = EigVec[:, EigVal.argsort()]
    pos_enc = torch.from_numpy(np.real(EigVec[:, 1:pos_enc_dim+1])).float()
    return pos_enc


def edges_from_note_array(note_array, gtype="heterogeneous"):
    edge_list, edge_types = sam.compute_edge_list(note_array['onset_div'].astype(np.int32),
                                                  note_array['duration_div'].astype(np.int32))
    if gtype == "heterogeneous":
        # concatenate edge types to edge list
        edge_list = np.concatenate((edge_list, edge_types.reshape(1, -1)), axis=0)
    return edge_list


def create_random_music_graph(graph_size, min_duration, max_duration, feature_size=10, add_beat_nodes=True):
    """
    Create a random score graph with random features

    The graph is created with 4 instruments, each with a random number of notes.

    Parameters
    ----------
    graph_size : int
        The number of nodes in the graph.
    min_duration : int
        The minimum duration of a note in the graph.
    max_duration : int
        The maximum duration of a note in the graph.
    """
    num_notes_per_voice = graph_size // 4
    dur = np.random.randint(min_duration, max_duration, size=(4, num_notes_per_voice))
    ons = np.cumsum(np.concatenate((np.zeros((4, 1)), dur), axis=1), axis=1)[:, :-1]
    dur = dur.flatten()
    ons = ons.flatten()

    pitch = np.row_stack((np.random.randint(70, 80, size=(1, num_notes_per_voice)),
                          np.random.randint(50, 60, size=(1, num_notes_per_voice)),
                          np.random.randint(60, 70, size=(1, num_notes_per_voice)),
                          np.random.randint(40, 50, size=(1, num_notes_per_voice)))).flatten()

    beats = ons / 4
    note_array = np.vstack((ons, dur, pitch, beats))
    # transform to structured array
    note_array = np.core.records.fromarrays(note_array, names='onset_div,duration_div,pitch,onset_beat')
    # create features array of shape (num_nodes, num_features)
    features = np.random.rand(len(note_array), feature_size)

    graph = create_score_graph(features, note_array, sort=True, add_reverse=True, add_beats=add_beat_nodes)
    return graph


def trim_to_layer(layer: int,
                  neighbor_mask_node:
                  torch.LongTensor,
                  neighbor_mask_edge: torch.LongTensor,
                  x: MaybeHeteroNodeTensor,
                  edge_index, # :MaybeHeteroEdgeTensor
                  edge_attr = None, # : Optional[MaybeHeteroEdgeTensor]
                  ):
    """Trims the :obj:`edge_index` representation, node features :obj:`x` and
    edge features :obj:`edge_attr` to a minimal-sized representation for the
    current GNN layer :obj:`layer` in directed
    :class:`~torch_geometric.loader.NeighborLoader` scenarios.

    This ensures that no computation is performed for nodes and edges that are
    not included in the current GNN layer, thus avoiding unnecessary
    computation within the GNN when performing neighborhood sampling.

    Args:
        layer (int): The current GNN layer.
        neighbor_mask_node (torch.LongTensor or Dict[NodeType, torch.LongTensor]): The
            mask of sampled nodes per hop.
        neighbor_mask_edge (torch.LongTensor or Dict[NodeType, torch.LongTensor]): The
            mask of sampled edges per hop.
        x (torch.Tensor or Dict[NodeType, torch.Tensor]): The homogeneous or
            heterogeneous (hidden) node features.
        edge_index (torch.Tensor or Dict[EdgeType, torch.Tensor]): The
            homogeneous or heterogeneous edge indices.
        edge_attr (torch.Tensor or Dict[EdgeType, torch.Tensor], optional): The
            homogeneous or heterogeneous (hidden) edge features.
    """
    if layer <= 0:
        return x, edge_index, edge_attr

    if isinstance(neighbor_mask_edge, dict):
        assert isinstance(neighbor_mask_node, dict)
        edge_mask = {k: v[v <= layer] for k, v in neighbor_mask_edge.items()}
        node_mask = {k: v[v <= layer] for k, v in neighbor_mask_node.items()}
        node_reindex = {k: torch.zeros_like(v) for k, v in node_mask.items()}
        for k, v in node_reindex.items():
            mask = node_mask[k] < layer
            node_reindex[k][mask] = torch.arange(mask.sum(), device=v.device)

        assert isinstance(x, dict)
        x = {k: v[node_mask[k] < layer] for k, v in x.items()}

        assert isinstance(edge_index, dict)
        edge_index = {
            k: v[:, edge_mask[k] < layer]
            for k, v in edge_index.items()
        }

        for k, v in edge_index.items():
            src_ntype = k[0]
            dst_ntype = k[-1]
            edge_index[k] = torch.stack([node_reindex[src_ntype][v[0]], node_reindex[dst_ntype][v[1]]], dim=0)

        if edge_attr is not None:
            assert isinstance(edge_attr, dict)
            edge_attr = {
                k: v[edge_mask[k] < layer]
                for k, v in edge_attr.items()
            }

        return x, edge_index, edge_attr

    assert isinstance(neighbor_mask_node, torch.LongTensor)
    neighbor_mask_edge = neighbor_mask_edge[neighbor_mask_edge <= layer]
    neighbor_mask_node = neighbor_mask_node[neighbor_mask_node <= layer]
    node_reindex = torch.empty_like(neighbor_mask_node)
    mask = neighbor_mask_node < layer
    node_reindex[mask] = torch.arange(mask.sum(), device=node_reindex.device)

    assert isinstance(x, torch.Tensor)
    x = x[neighbor_mask_node < layer]

    assert isinstance(edge_index, (torch.Tensor, SparseTensor))
    edge_index = edge_index[:, neighbor_mask_edge < layer]
    edge_index = torch.stack([node_reindex[edge_index[0]], node_reindex[edge_index[1]]], dim=0)

    if edge_attr is not None:
        assert isinstance(edge_attr, torch.Tensor)
        edge_attr = edge_attr[neighbor_mask_edge < layer]

    return x, edge_index, edge_attr