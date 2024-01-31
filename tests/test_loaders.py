from graphmuse.loader import MuseNeighborLoader
import numpy as np
from graphmuse.samplers import compute_edge_list, c_set_seed
import torch
from torch_geometric.data import HeteroData

# Standardize the random seed
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
c_set_seed(42)
torch.backends.cudnn.deterministic = True

num_graphs = 10
max_nodes = 500
min_nodes = 100
max_dur = 20
min_dur = 1
subgraph_size = 100
subgraphs = 10

graphs = list()
for i in range(num_graphs):
    l = np.random.randint(min_nodes, max_nodes)
    dur = np.random.randint(min_dur, max_dur, size=(4, l))
    ons = np.cumsum(np.concatenate((np.zeros((4, 1)), dur), axis=1), axis=1)[:, :-1]
    dur = dur.flatten()
    ons = ons.flatten()

    pitch = np.row_stack((np.random.randint(70, 80, size=(1, l)),
                          np.random.randint(50, 60, size=(1, l)),
                          np.random.randint(60, 70, size=(1, l)),
                          np.random.randint(40, 50, size=(1, l)))).flatten()
    note_array = np.vstack((ons, dur, pitch))
    # transform to structured array
    note_array = np.core.records.fromarrays(note_array, names='onset_div,duration_div,pitch')
    # sort by onset and then by pitch
    note_array = np.sort(note_array, order=['onset_div', 'pitch'])
    edges, edge_types = compute_edge_list(note_array['onset_div'].astype(np.int32),
                                          note_array['duration_div'].astype(np.int32))

    # sort edges
    resort_idx = np.lexsort((edges[0], edges[1]))
    edges = edges[:, resort_idx]

    edge_types = edge_types[resort_idx]
    # create features
    features = np.random.rand(note_array.shape[0], 10)
    # create graph
    # new_edges = np.vstack((edges, edge_types))
    edge_etypes = {
        0: "onset",
        1: "consecutive",
        2: "duration",
        3: "rest"
    }
    edges = torch.from_numpy(edges).long()
    edge_types = torch.from_numpy(edge_types).long()
    graph = HeteroData()
    graph["note"].x = torch.from_numpy(features).float()
    graph["note"].onset_div = torch.from_numpy(note_array['onset_div']).long()
    graph["note"].duration_div = torch.from_numpy(note_array['duration_div']).long()
    graph["note"].pitch = torch.from_numpy(note_array['pitch']).long()
    for k, v in edge_etypes.items():
        graph['note', v, 'note'].edge_index = edges[:, edge_types == k]
    graphs.append(graph)

# create dataloader
dataloader = MuseNeighborLoader(graphs, subgraph_size=50, batch_size=4, num_neighbors=[3, 3])
batch = next(iter(dataloader))
print(batch)