import numpy as np
import torch
import warnings
import graphmuse.samplers as csamplers
import partitura.score as spt
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from typing import Optional, Union, Tuple, List, Dict, Any


def graph_to_pyg(x, edge_index, edge_attributes=None, note_array=None):
    """
    Convert graph data to PyTorch Geometric HeteroData format.

    Parameters
    ----------
    x : np.ndarray or torch.Tensor
        Node features.
    edge_index : np.ndarray
        Edge indices.
    edge_attributes : np.ndarray, optional
        Edge attributes, by default None.
    note_array : np.ndarray, optional
        Note array, by default None.

    Returns
    -------
    HeteroData
        PyTorch Geometric HeteroData object.
    """
    edge_type_map = {"onset": 0, "consecutive": 1, "during": 2, "rest": 3}
    data = HeteroData()
    data["note"].x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
    edge_type = torch.from_numpy(edge_index[2])
    edge_index = torch.from_numpy(edge_index[:2])
    for k, v in edge_type_map.items():
        data["note", k, "note"].edge_index = torch.from_numpy(edge_index[:, edge_type == v])
        if edge_attributes is not None:
            data["note", k, "note"].edge_attr = torch.from_numpy(edge_attributes[edge_type == v])
    if note_array is not None:
        for k in note_array.dtype.names:
            data["note", k] = torch.from_numpy(note_array[k])
    return data


def edges_from_note_array(note_array):
    """
    Turn note_array to list of edges.

    Parameters
    ----------
    note_array : structured array
        The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.

    Returns
    -------
    np.ndarray
        The edges in the shape of (2, num_edges).
    """
    edg_src = list()
    edg_dst = list()
    start_rest_index = len(note_array)
    for i, x in enumerate(note_array):
        for j in np.where((note_array["onset_div"] == x["onset_div"]))[0]: #& (note_array["id"] != x["id"]))[0]:
            edg_src.append(i)
            edg_dst.append(j)

        for j in np.where(note_array["onset_div"] == x["onset_div"] + x["duration_div"])[0]:
            edg_src.append(i)
            edg_dst.append(j)

        for j in np.where((x["onset_div"] < note_array["onset_div"]) & (x["onset_div"] + x["duration_div"] > note_array["onset_div"]))[0]:
            edg_src.append(i)
            edg_dst.append(j)

    end_times = note_array["onset_div"] + note_array["duration_div"]
    for et in np.sort(np.unique(end_times))[:-1]:
        if et not in note_array["onset_div"]:
            scr = np.where(end_times == et)[0]
            diffs = note_array["onset_div"] - et
            tmp = np.where(diffs > 0, diffs, np.inf)
            dst = np.where(tmp == tmp.min())[0]
            for i in scr:
                for j in dst:
                    edg_src.append(i)
                    edg_dst.append(j)

    edges = np.array([edg_src, edg_dst])
    return edges


def add_reverse_edges(graph, mode):
    """
    Add reverse edges to the graph.

    Parameters
    ----------
    graph : HeteroData
        The graph object.
    mode : str
        The mode of adding reverse edges. Either 'new_type' or 'undirected'.

    Returns
    -------
    HeteroData
        The graph with added reverse edges.
    """
    if mode == "new_type":
        # add reversed consecutive edges
        graph["note", "consecutive_rev", "note"].edge_index = graph[
            "note", "consecutive", "note"
        ].edge_index[[1, 0]]
        # add reversed during edges
        graph["note", "during_rev", "note"].edge_index = graph[
            "note", "during", "note"
        ].edge_index[[1, 0]]
        # add reversed rest edges
        graph["note", "rest_rev", "note"].edge_index = graph[
            "note", "rest", "note"
        ].edge_index[[1, 0]]
    elif mode == "undirected":
        graph = ToUndirected()(graph)
    else:
        raise ValueError("mode must be either 'new_type' or 'undirected'")
    return graph


def add_reverse_edges_from_edge_index(edge_index, edge_type, mode="new_type"):
    """
    Add reverse edges to the edge index.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge indices.
    edge_type : torch.Tensor
        Edge types.
    mode : str, optional
        The mode of adding reverse edges, by default "new_type".

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Updated edge indices and edge types.
    """
    if mode == "new_type":
        unique_edge_types = torch.unique(edge_type)
        for type in unique_edge_types:
            if type == 0:
                continue
            edge_index = torch.cat((edge_index, edge_index[:, edge_type == type].flip(0)), dim=1)
            edge_type = torch.cat((edge_type, torch.max(edge_type) + torch.zeros(edge_index.shape[1] - edge_type.shape[0], dtype=torch.long).to(edge_type.device)), dim=0)
    else:
        edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
        edge_type = torch.cat((edge_type, edge_type), dim=0)
    return edge_index, edge_type


def add_measure_nodes(measures, note_array):
    """
    Add virtual nodes for every measure.

    Parameters
    ----------
    measures : np.ndarray or list
        Measures data.
    note_array : np.ndarray
        Note array.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int]
        Measure cluster, measure edges, and number of measures.
    """
    assert "onset_div" in note_array.dtype.names, "Note array must have 'onset_div' field to add measure nodes."
    if not isinstance(measures, np.ndarray):
        measures = np.array([[m.start.t, m.end.t] for m in measures])

    onset_div = note_array["onset_div"]
    measure_cluster = np.zeros(len(note_array), dtype=np.int64) - 1
    edges = []

    m_ptr = 0  # pointer for measures
    o_ptr = 0  # pointer for onset_div

    while m_ptr < len(measures) and o_ptr < len(onset_div):
        if onset_div[o_ptr] < measures[m_ptr][1]:
            measure_cluster[o_ptr] = m_ptr
            edges.append([o_ptr, m_ptr])
            o_ptr += 1
        else:
            m_ptr += 1

    num_measures = len(measures)
    measure_edges = np.array(edges).T
    # Warn if all edges is empty
    if measure_edges.size == 0:
        warnings.warn(
            f"No edges found for measure nodes. Check that the note array has the 'onset_div' field on score.")

    # Verify that every note has a measure
    assert np.all(measure_cluster != -1), "Not all notes have a measure."

    return measure_cluster, measure_edges, num_measures


def add_beat_nodes(note_array):
    """
    Add virtual nodes for every measure.

    Parameters
    ----------
    note_array : np.ndarray
        Note array.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Beat cluster, beat index, and beat edges.
    """
    assert "onset_beat" in note_array.dtype.names, "Note array must have 'onset_beat' field to add beat nodes."
    # when the onset_beat has negative values, we need to shift all the values to be positive
    onset_beat = note_array["onset_beat"]
    # beat_edges, beat_index, beat_cluster = csamplers.compute_beat_edges(onset_beat)
    if onset_beat.min() < 0:
        onset_beat = onset_beat - onset_beat.min()

    beat_index = np.arange(int(onset_beat.max())+1)
    beat_cluster = np.zeros(len(onset_beat), dtype=np.int64) - 1
    # Create an empty array of edges size onset_beat x 2
    edges = np.zeros((len(onset_beat), 2), dtype=np.int64)

    b_ptr = 0  # pointer for beat_index
    o_ptr = 0  # pointer for onset_beat

    while b_ptr < len(beat_index) and o_ptr < len(onset_beat):
        if onset_beat[o_ptr] < beat_index[b_ptr] + 1:
            edges[o_ptr][0] = o_ptr
            edges[o_ptr][1] = b_ptr
            beat_cluster[o_ptr] = b_ptr
            o_ptr += 1
        else:
            b_ptr += 1

    beat_edges = edges.T

    # Verify that every note has a beat
    assert np.all(beat_cluster != -1), "Not all notes have a beat."

    return beat_cluster, beat_index, beat_edges


def create_score_graph(
        features: Union[np.ndarray, torch.Tensor],
        note_array: np.ndarray,
        labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        sort: bool=False,
        add_reverse: bool= True,
        measures: Optional[List[spt.Measure]] = None,
        add_beats: bool = False) -> HeteroData:
    """
    Create a score graph from note array.

    Parameters
    ----------
    features : Union[np.ndarray, torch.Tensor]
        The feature matrix, in the shape of (num_nodes, num_features).
    note_array : np.ndarray
        The note array object, it is a structured array and needs to contain the following fields:
        onset_div, duration_div, pitch.
    labels : Union[np.ndarray, torch.Tensor], optional
        The labels for the note nodes, by default None.
    sort : bool, optional
        Whether to sort the note array, by default False.
    add_reverse : bool, optional
        Whether to add reverse edges, by default True.
    measures : Union[Optional, List[spt.Measure]], optional
        The measure objects, by default None.
    add_beats : bool, optional
        Whether to add beat nodes, by default False.

    Returns
    -------
    HeteroData
        The score graph.
    """
    if sort:
        note_array = np.sort(note_array, order=["onset_div", "pitch"])

    edges, edge_types = csamplers.compute_edge_list(
        note_array['onset_div'].astype(np.int32),
        note_array['duration_div'].astype(np.int32))

    # create graph
    # new_edges = np.vstack((edges, edge_types))
    edge_etypes = {
        0: "onset",
        1: "consecutive",
        2: "during",
        3: "rest"
    }
    edges = torch.from_numpy(edges).long()
    edge_types = torch.from_numpy(edge_types).long()
    graph = HeteroData()
    graph["note"].x = torch.from_numpy(features).float() if isinstance(features, np.ndarray) else features.float()
    if labels is not None:
        graph["note"].y = torch.from_numpy(labels).long() if isinstance(labels, np.ndarray) else labels.long()
    graph["note"].onset_div = torch.from_numpy(note_array['onset_div'].copy()).long()
    graph["note"].duration_div = torch.from_numpy(note_array['duration_div'].copy()).long()
    graph["note"].pitch = torch.from_numpy(note_array['pitch'].copy()).long()
    for k, v in edge_etypes.items():
        graph['note', v, 'note'].edge_index = edges[:, edge_types == k]

    if add_reverse:
        graph = add_reverse_edges(graph, mode="new_type")

    if measures is not None:
        measure_cluster, measure_edges, num_measures = add_measure_nodes(measures, note_array)
        graph["note"].measure_cluster = torch.from_numpy(measure_cluster).long()
        graph["note", "connects", "measure"].edge_index = torch.from_numpy(measure_edges)
        graph["measure", "connects", "note"].edge_index = graph["note", "connects", "measure"].edge_index.flip(0)
        measure_index = torch.arange(num_measures).long()
        graph["measure", "next", "measure"].edge_index = torch.vstack((measure_index[:-1], measure_index[1:]))
        # scatter note_features to measure_features based on measure_cluster
        graph["measure"].index = measure_index
        graph["measure"].x = torch.zeros(measure_index.shape[0], graph["note"].x.shape[1])
        graph["measure"].x.scatter_add_(0, graph["note"].measure_cluster.unsqueeze(-1).expand(-1, graph["note"].x.shape[1]), graph["note"].x)

    if add_beats:
        beat_cluster, beat_index, beat_edges = add_beat_nodes(note_array)
        graph["note"].beat_cluster = torch.from_numpy(beat_cluster).long()
        graph["note", "connects", "beat"].edge_index = torch.from_numpy(beat_edges)
        graph["beat", "connects", "note"].edge_index = graph["note", "connects", "beat"].edge_index.flip(0)
        beat_index = torch.from_numpy(beat_index).long()
        graph["beat", "next", "beat"].edge_index = torch.vstack((beat_index[:-1], beat_index[1:]))
        # scatter note_features to beat_features based on beat_cluster
        graph["beat"].index = beat_index
        graph["beat"].x = torch.zeros(beat_index.shape[0], graph["note"].x.shape[1])
        graph["beat"].x.scatter_add_(0, graph["note"].beat_cluster.unsqueeze(-1).expand(-1, graph["note"].x.shape[1]), graph["note"].x)

    return graph
