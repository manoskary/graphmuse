from graphmuse.samplers.sampling_sketch import MuseDataloader
from graphmuse.utils.graph import edges_from_note_array, HeteroScoreGraph
import numpy as np


num_graphs = 100
max_nodes = 5000
min_nodes = 500
max_dur = 20
min_dur = 1
subgraph_size = 100
subgraphs = 10

graphs = list()
for i in range(num_graphs):
    l = np.random.randint(min_nodes, max_nodes)
    dur = np.random.randint(min_dur, max_dur, size=(4, l))
    ons = np.cumsum(np.concatenate((np.zeros((4, 1)), dur), axis=1), axis=1)[:,:-1]
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
    edges = edges_from_note_array(note_array)
    # sort edges
    edges = edges[:, edges[0].argsort()]
    # create features
    features = np.random.rand(note_array.shape[0], 10)
    # create graph
    graph = HeteroScoreGraph(features, edges, note_array=note_array)
    graphs.append(graph)

# create dataloader
dataloader = MuseDataloader(graphs, subgraph_size=100, subgraphs=10, batch_size=1, num_workers=0)
# iterate over dataloader
batch = next(iter(dataloader))


