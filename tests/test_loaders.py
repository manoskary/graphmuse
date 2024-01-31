from graphmuse.loader import MuseNeighborLoader
import numpy as np
from graphmuse.samplers import c_set_seed
from graphmuse.utils import create_score_graph
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
batch_size = 10

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
    # create features array of shape (num_nodes, num_features)
    features = np.random.rand(len(note_array), 10)
    graph = create_score_graph(features, note_array, sort=True, add_reverse=True)
    graphs.append(graph)

# create dataloader
dataloader = MuseNeighborLoader(graphs, subgraph_size=subgraph_size, batch_size=batch_size, num_neighbors=[3, 3])
batch = next(iter(dataloader))
print(batch)
