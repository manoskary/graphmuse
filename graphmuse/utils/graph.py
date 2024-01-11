import numpy as np
# from cython_graph import GraphFromAdj
import torch
from numpy.lib import recfunctions as rfn
import os
import random
import string
import pickle
from .general import MapDict
import warnings
import graphmuse.samplers as csamplers
from torch_geometric.data import HeteroData


class HeteroScoreGraph(object):
    """
    Heterogeneous Score Graph.

    Parameters
    ----------
    note_features : numpy array or torch tensor
        The features of the notes in the score.
    edges : numpy
        The edges of the score graph.
    etypes : list of str
        The types of the edges.
    name : str
        The name of the score graph.
    note_array : structured numpy array (optional)
        The note array of the score.
    edge_weights : numpy array (optional)
        The weights of the edges.
    labels : numpy array (optional)
        The labels of the notes.
    """

    def __init__(self, note_features, edges, etypes=["onset", "consecutive", "during", "rest"], name=None,
                 note_array=None, edge_weights=None, labels=None):
        self.node_features = note_features.dtype.names if note_features.dtype.names else []
        self.features = note_features
        # Filter out string fields of structured array.
        if self.node_features:
            self.node_features = [feat for feat in self.node_features if
                                  note_features.dtype.fields[feat][0] != np.dtype('U256')]
            self.features = self.features[self.node_features]
        self.x = torch.from_numpy(
            np.asarray(rfn.structured_to_unstructured(self.features) if self.node_features else self.features,
                       dtype=np.float32))
        assert etypes is not None
        self.etypes = {t: i for i, t in enumerate(etypes)}
        self.note_array = note_array
        self.edge_type = torch.from_numpy(edges[-1].astype(np.int32)).long()
        self.edge_index = torch.from_numpy(edges[:2].astype(np.int32)).long()
        self.c_graph = csamplers.graph(edges[:2])
        self.edge_weights = torch.ones(len(self.edge_index[0])) if edge_weights is None else torch.from_numpy(
            edge_weights)
        self.name = name
        self.y = labels if labels is None else torch.from_numpy(labels)

    def node_count(self):
        return len(self.note_array)

    def adj(self, weighted=False):
        if weighted:
            return torch.sparse_coo_tensor(self.edge_index, self.edge_weights, (len(self.x), len(self.x)))
        ones = torch.ones(len(self.edge_index[0]))
        matrix = torch.sparse_coo_tensor(self.edge_index, ones, (len(self.x), len(self.x)))
        return matrix

    def add_measure_nodes(self, measures):
        """Add virtual nodes for every measure"""
        assert "onset_div" in self.note_array.dtype.names, "Note array must have 'onset_div' field to add measure nodes."
        if not isinstance(measures, np.ndarray):
            measures = np.array([[m.start.t, m.end.t] for m in measures])
        # if not hasattr(self, "beat_nodes"):
        #     self.add_beat_nodes()
        nodes = np.arange(len(measures))
        # Add new attribute to hg
        edges = []
        for i in range(len(measures)):
            idx = np.where(
                (self.note_array["onset_div"] >= measures[i, 0]) & (self.note_array["onset_div"] < measures[i, 1]))[0]
            if idx.size:
                edges.append(np.vstack((idx, np.full(idx.size, i))))
        self.measure_nodes = nodes
        self.measure_edges = np.hstack(edges)
        # Warn if all edges is empty
        if self.measure_edges.size == 0:
            warnings.warn(
                f"No edges found for measure nodes. Check that the note array has the 'onset_div' field on score {self.name}.")

    def add_beat_nodes(self):
        """Add virtual nodes for every beat"""
        assert "onset_beat" in self.note_array.dtype.names, "Note array must have 'onset_beat' field to add measure nodes."
        nodes = np.arange(int(self.note_array["onset_beat"].max()))
        # Add new attribute to hg

        edges = []
        for b in nodes:
            idx = np.where((self.note_array["onset_beat"] >= b) & (self.note_array["onset_beat"] < b + 1))[0]
            if idx.size:
                edges.append(np.vstack((idx, np.full(idx.size, b))))
        self.beat_nodes = nodes
        self.beat_edges = np.hstack(edges)

    def assign_typed_weight(self, weight_dict: dict):
        assert weight_dict.keys() == self.etypes.keys()
        for k, v in weight_dict.items():
            etype = self.etypes[k]
            self.edge_weights[self.edge_type == etype] = v

    def get_edges_of_type(self, etype):
        assert etype in self.etypes.keys()
        etype = self.etypes[etype]
        return self.edge_index[:, self.edge_type == etype]

    def save(self, save_dir):
        save_name = self.name if self.name else ''.join(random.choice(string.ascii_letters) for i in range(10))
        (os.makedirs(os.path.join(save_dir, save_name)) if not os.path.exists(
            os.path.join(save_dir, save_name)) else None)
        object_properties = vars(self)
        with open(os.path.join(save_dir, save_name, "x.npy"), "wb") as f:
            np.save(f, self.x.numpy())
        del object_properties['x']
        with open(os.path.join(save_dir, save_name, "edge_index.npy"), "wb") as f:
            np.save(f, torch.cat((self.edge_index, self.edge_type.unsqueeze(0))).numpy())
        del object_properties['edge_index']
        del object_properties['edge_type']
        if isinstance(self.y, torch.Tensor):
            with open(os.path.join(save_dir, save_name, "y.npy"), "wb") as f:
                np.save(f, self.y.numpy())
            del object_properties['y']
        if isinstance(self.edge_weights, torch.Tensor):
            np.save(open(os.path.join(save_dir, save_name, "edge_weights.npy"), "wb"), self.edge_weights.numpy())
            del object_properties['edge_weights']
        if isinstance(self.note_array, np.ndarray):
            np.save(open(os.path.join(save_dir, save_name, "note_array.npy"), "wb"), self.note_array)
            del object_properties['note_array']
        with open(os.path.join(save_dir, save_name, 'graph_info.pkl'), 'wb') as handle:
            pickle.dump(object_properties, handle, protocol=pickle.HIGHEST_PROTOCOL)



def graph_to_pyg(x, edge_index, edge_attributes=None, note_array=None):
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


def load_score_hgraph(load_dir, name=None):
    path = os.path.join(load_dir, name) if os.path.basename(load_dir) != name else load_dir
    if not os.path.exists(path) or not os.path.isdir(path):
        raise ValueError("The directory is not recognized.")
    x = np.load(open(os.path.join(path, "x.npy"), "rb"))
    edge_index = np.load(open(os.path.join(path, "edge_index.npy"), "rb"))
    graph_info = pickle.load(open(os.path.join(path, "graph_info.pkl"), "rb"))
    y = np.load(open(os.path.join(path, "y.npy"), "rb")) if os.path.exists(os.path.join(path, "y.npy")) else None
    edge_weights = np.load(open(os.path.join(path, "edge_weights.npy"), "rb")) if os.path.exists(os.path.join(path, "edge_weights.npy")) else None
    note_array = np.load(open(os.path.join(path, "note_array.npy"), "rb")) if os.path.exists(
        os.path.join(path, "note_array.npy")) else None
    name = name if name else os.path.basename(path)
    hg = HeteroScoreGraph(note_features=x, edges=edge_index, name=name, labels=y, edge_weights=edge_weights, note_array=note_array)
    for k, v in graph_info.items():
        setattr(hg, k, v)
    return hg


def load_score_hgraph(load_dir, name=None):
    path = os.path.join(load_dir, name) if os.path.basename(load_dir) != name else load_dir
    if not os.path.exists(path) or not os.path.isdir(path):
        raise ValueError("The directory is not recognized.")
    x = np.load(open(os.path.join(path, "x.npy"), "rb"))
    edge_index = np.load(open(os.path.join(path, "edge_index.npy"), "rb"))
    graph_info = pickle.load(open(os.path.join(path, "graph_info.pkl"), "rb"))
    y = np.load(open(os.path.join(path, "y.npy"), "rb")) if os.path.exists(os.path.join(path, "y.npy")) else None
    y = graph_info.y if hasattr(graph_info, "y") and y is None else y
    edge_weights = np.load(open(os.path.join(path, "edge_weights.npy"), "rb")) if os.path.exists(os.path.join(path, "edge_weights.npy")) else None
    note_array = np.load(open(os.path.join(path, "note_array.npy"), "rb")) if os.path.exists(
        os.path.join(path, "note_array.npy")) else None
    name = name if name else os.path.basename(path)
    hg = HeteroScoreGraph(note_features=x, edges=edge_index, name=name, labels=y, edge_weights=edge_weights, note_array=note_array)
    for k, v in graph_info.items():
        setattr(hg, k, v)
    return hg


def check_note_array(na):
    dtypes = na.dtype.names
    if not all([x in dtypes for x in ["onset_beat", "duration_beat", "ts_beats", "ts_beat_type"]]):
        raise(TypeError("The given Note array is missing necessary fields."))


class BatchedHeteroScoreGraph(HeteroScoreGraph):
    def __init__(self, note_features, edges, lengths, etypes=["onset", "consecutive", "during", "rest"], name=None, note_array=None, edge_weights=None, labels=None):
        super(BatchedHeteroScoreGraph, self).__init__(note_features, edges, etypes, name, note_array, edge_weights, labels)
        self.lengths = lengths

    def unbatch(self):
        graphs = []
        for i, l in enumerate(self.lengths):
            graphs.append(HeteroScoreGraph(self.x[i, :l], self.edge_index[i, :, :l], self.etypes, self.name, self.note_array[i, :l], self.edge_weights[i, :l], self.y[i]))
        return graphs


def batch_graphs(graphs):
    """
    Batch a list of graphs into a single graph.

    Returns:
    --------
    batched_graph: HeteroScoreGraph
        A single graph with the same attributes as the input graphs, but with
        batched attributes.
    """
    lengths = [0] + [len(g.x) for g in graphs[:-1]]
    new_edges = np.concatenate([g.edge_index + lengths[i] for i, g in enumerate(graphs)], axis=1)
    batched_graph = BatchedHeteroScoreGraph(note_features=np.concatenate([g.x.numpy() for g in graphs], axis=0),
                                     edges=new_edges,
                                     lengths=lengths,
                                     etypes=graphs[0].etypes,
                                     names=[g.name for g in graphs],
                                     labels=np.concatenate([g.y.numpy() for g in graphs], axis=0) if graphs[0].y is not None else None,
                                     edge_weights=np.concatenate([g.edge_weights.numpy() for g in graphs], axis=0) if graphs[0].edge_weights is not None else None,
                                     note_array=np.concatenate([g.note_array for g in graphs], axis=0) if graphs[0].note_array is not None else None)
    return batched_graph


def edges_from_note_array(note_array):
    '''Turn note_array to list of edges.

    Parameters
    ----------
    note_array : structured array
        The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.

    Returns
    -------
    edg_src : np.array
        The edges in the shape of (2, num_edges).
    '''

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


def graph_from_notearray(note_array, cython=True):
    if cython:
        edges = GraphFromAdj(
            np.ascontiguousarray(note_array["onset_beat"], np.float32),
            np.ascontiguousarray(note_array["duration_beat"], np.float32),
            np.ascontiguousarray(note_array["pitch"], np.float32),
            4).process()
    else:
        edges = graph_from_note_array(note_array)
    return edges


def add_reverse_edges(graph, mode):
    if isinstance(graph, HeteroScoreGraph):
        if mode == "new_type":
            # Add reverse During Edges
            graph.edge_index = torch.cat((graph.edge_index, graph.get_edges_of_type("during").flip(0)), dim=1)
            graph.edge_type = torch.cat((graph.edge_type, 2 + torch.zeros(graph.edge_index.shape[1] - graph.edge_type.shape[0],dtype=torch.long)), dim=0)
            # Add reverse Consecutive Edges
            graph.edge_index = torch.cat((graph.edge_index, graph.get_edges_of_type("consecutive").flip(0)), dim=1)
            graph.edge_type = torch.cat((graph.edge_type, 4+torch.zeros(graph.edge_index.shape[1] - graph.edge_type.shape[0], dtype=torch.long)), dim=0)
            graph.etypes["consecutive_rev"] = 4
        else:
            graph.edge_index = torch.cat((graph.edge_index, graph.edge_index.flip(0)), dim=1)
            raise NotImplementedError("To undirected is not Implemented for HeteroScoreGraph.")
    # elif isinstance(graph, ScoreGraph):
    #     raise NotImplementedError("To undirected is not Implemented for ScoreGraph.")
    else:
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
            graph = pyg.transforms.ToUndirected()(graph)
        else:
            raise ValueError("mode must be either 'new_type' or 'undirected'")
    return graph


def add_reverse_edges_from_edge_index(edge_index, edge_type, mode="new_type"):
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



def node_subgraph(graph, nodes, include_measures=False):
    """
    Extract subgraph given a list of node indices.

    Parameters
    ----------
    graph : dict or HeteroScoreGraph
    nodes : torch.Tensor
        List of node indices.

    Returns
    -------
    out : dict
    """
    out = dict()
    graph = MapDict(graph) if isinstance(graph, dict) else graph
    assert torch.arange(graph.x.shape[0]).max() >= nodes.max(), "Node indices must be smaller than the number of nodes in the graph."
    nodes_min = nodes.min()
    edge_indices = torch.isin(graph.edge_index[0], nodes) & torch.isin(
        graph.edge_index[1], nodes)
    out["x"] = graph.x[nodes]
    out["edge_index"] = graph.edge_index[:, edge_indices] - nodes_min
    out["y"] = graph.y[nodes] if graph.y.shape[0] == graph.x.shape[0] else graph.y
    out["edge_type"] = graph.edge_type[edge_indices]
    out["note_array"] = structured_to_unstructured(
        graph.note_array[
            ["pitch", "onset_div", "duration_div", "onset_beat", "duration_beat", "ts_beats"]]
    )[indices] if isinstance(graph, HeteroScoreGraph) else graph.note_array[nodes]
    out["name"] = graph.name
    if include_measures:
        measure_edges = torch.tensor(graph.measure_edges)
        measure_nodes = torch.tensor(graph.measure_nodes).squeeze()
        beat_edges = torch.tensor(graph.beat_edges)
        beat_nodes = torch.tensor(graph.beat_nodes).squeeze()
        beat_edge_indices = torch.isin(beat_edges[0], nodes)
        beat_node_indices = torch.isin(beat_nodes, torch.unique(beat_edges[1][beat_edge_indices]))
        min_beat_idx = torch.where(beat_node_indices)[0].min()
        max_beat_idx = torch.where(beat_node_indices)[0].max()
        measure_edge_indices = torch.isin(measure_edges[0], nodes)
        measure_node_indices = torch.isin(measure_nodes, torch.unique(measure_edges[1][measure_edge_indices]))
        min_measure_idx = torch.where(measure_node_indices)[0].min()
        max_measure_idx = torch.where(measure_node_indices)[0].max()
        out["beat_nodes"] = beat_nodes[min_beat_idx:max_beat_idx + 1] - min_beat_idx
        out["beat_edges"] = torch.vstack(
            (beat_edges[0, beat_edge_indices] - nodes_min, beat_edges[1, beat_edge_indices] - min_beat_idx))
        out["measure_nodes"] = measure_nodes[min_measure_idx:max_measure_idx + 1] - min_measure_idx
        out["measure_edges"] = torch.vstack((measure_edges[0, measure_edge_indices] - nodes_min,
                                             measure_edges[1, measure_edge_indices] - min_measure_idx))
    return out


# if __name__ == '__main__':
#     import partitura as pt
#     from timeit import default_timer as timer
#
#     score_dir = "/home/manos/Desktop/JKU/data/test.musicxml"
#     note_array = pt.load_score(score_dir).note_array()
#
#     # lst = []
#     # for i in range(10):
#     #     start = timer()
#     #     graph_from_notearray(note_array, cython=False)
#     #     end = timer()
#     #     lst.append(end-start)
#     # print("Python Code: ", sum(lst) / len(lst))
#
#     lst = []
#     for i in range(10):
#         start = timer()
#         graph_from_notearray(note_array, cython=True)
#         end = timer()
#         lst.append(end - start)
#     print("Cython Code: ", sum(lst) / len(lst))