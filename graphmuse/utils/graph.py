import numpy as np
from cython_graph import GraphFromAdj
import torch
from numpy.lib import recfunctions as rfn
import os
import random
import string
import pickle



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
    def __init__(self, note_features, edges, etypes=["onset", "consecutive", "during", "rest"], name=None, note_array=None, edge_weights=None, labels=None):
        self.node_features = note_features.dtype.names if note_features.dtype.names else []
        self.features = note_features
        # Filter out string fields of structured array.
        if self.node_features:
            self.node_features = [feat for feat in self.node_features if note_features.dtype.fields[feat][0] != np.dtype('U256')]
            self.features = self.features[self.node_features]
        self.x = torch.from_numpy(np.asarray(rfn.structured_to_unstructured(self.features) if self.node_features else self.features))
        assert etypes is not None
        self.etypes = {t: i for i, t in enumerate(etypes)}
        self.note_array = note_array
        self.edge_type = torch.from_numpy(edges[-1]).long()
        self.edge_index = torch.from_numpy(edges[:2]).long()
        self.edge_weights = torch.ones(len(self.edge_index[0])) if edge_weights is None else torch.from_numpy(edge_weights)
        self.name = name
        self.y = labels if labels is None else torch.from_numpy(labels).long()

    def adj(self, weighted=False):
        if weighted:
            return torch.sparse_coo_tensor(self.edge_index, self.edge_weights, (len(self.x), len(self.x)))
        ones = torch.ones(len(self.edge_index[0]))
        matrix = torch.sparse_coo_tensor(self.edge_index, ones, (len(self.x), len(self.x)))
        return matrix

    def assign_typed_weight(self, weight_dict:dict):
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
        (os.makedirs(os.path.join(save_dir, save_name)) if not os.path.exists(os.path.join(save_dir, save_name)) else None)
        with open(os.path.join(save_dir, save_name, "x.npy"), "wb") as f:
            np.save(f, self.x.numpy())
        with open(os.path.join(save_dir, save_name, "edge_index.npy"), "wb") as f:
            np.save(f, torch.cat((self.edge_index, self.edge_type.unsqueeze(0))).numpy())
        if isinstance(self.y, torch.Tensor):
            with open(os.path.join(save_dir, save_name, "y.npy"), "wb") as f:
                np.save(f, self.y.numpy())
        if isinstance(self.edge_weights, torch.Tensor):
            np.save(open(os.path.join(save_dir, save_name, "edge_weights.npy"), "wb"), self.edge_weights.numpy())
        if isinstance(self.note_array, np.ndarray):
            np.save(open(os.path.join(save_dir, save_name, "note_array.npy"), "wb"), self.note_array)
        with open(os.path.join(save_dir, save_name, 'graph_info.pkl'), 'wb') as handle:
            object_properties = vars(self)
            del object_properties['x']
            del object_properties['edge_index']
            del object_properties['edge_type']
            del object_properties['edge_weights']
            del object_properties['y']
            del object_properties['note_array']
            pickle.dump(object_properties, handle, protocol=pickle.HIGHEST_PROTOCOL)


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



def graph_from_note_array(note_array, rest_array=None, norm2bar=True):
    '''Turn note_array to homogeneous graph dictionary.
    Parameters
    ----------
    note_array : structured array
        The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.
    rest_array : structured array
        A structured rest array similar to the note array but for rests.
    t_sig : list
        A list of time signature in the piece.
    '''

    edg_src = list()
    edg_dst = list()
    start_rest_index = len(note_array)
    for i, x in enumerate(note_array):
        for j in np.where((np.isclose(note_array["onset_beat"], x["onset_beat"], rtol=1e-04, atol=1e-04) == True) & (note_array["pitch"] != x["pitch"]))[0]:
            edg_src.append(i)
            edg_dst.append(j)

        for j in np.where(np.isclose(note_array["onset_beat"], x["onset_beat"] + x["duration_beat"], rtol=1e-04, atol=1e-04) == True)[0]:
            edg_src.append(i)
            edg_dst.append(j)


        for j in np.where((x["onset_beat"] < note_array["onset_beat"]) & (x["onset_beat"] + x["duration_beat"] > note_array["onset_beat"]))[0]:
            edg_src.append(i)
            edg_dst.append(j)

    end_times = note_array["onset_beat"] + note_array["duration_beat"]
    for et in np.sort(np.unique(end_times))[:-1]:
        if et not in note_array["onset_beat"]:
            scr = np.where(end_times == et)[0]
            diffs = note_array["onset_beat"] - et
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


if __name__ == '__main__':
    import partitura as pt
    from timeit import default_timer as timer

    score_dir = "/home/manos/Desktop/JKU/data/test.musicxml"
    note_array = pt.load_score(score_dir).note_array()

    # lst = []
    # for i in range(10):
    #     start = timer()
    #     graph_from_notearray(note_array, cython=False)
    #     end = timer()
    #     lst.append(end-start)
    # print("Python Code: ", sum(lst) / len(lst))

    lst = []
    for i in range(10):
        start = timer()
        graph_from_notearray(note_array, cython=True)
        end = timer()
        lst.append(end - start)
    print("Cython Code: ", sum(lst) / len(lst))