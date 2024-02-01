from graphmuse.loader import MuseNeighborLoader
import numpy as np
from graphmuse.samplers import c_set_seed
from graphmuse.utils import create_random_music_graph
import torch
from unittest import TestCase

# Standardize the random seed
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
c_set_seed(42)
torch.backends.cudnn.deterministic = True


class TestMuseNeighborLoader(TestCase):
    def test_muse_neighbor_loader(self):
        # Create a random graph
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
            graph = create_random_music_graph(
                graph_size=l, min_duration=min_dur, max_duration=max_dur, add_beat_nodes=True)
            graphs.append(graph)

        # create dataloader
        dataloader = MuseNeighborLoader(graphs, subgraph_size=subgraph_size, batch_size=batch_size,
                                        num_neighbors=[3, 3])
        batch = next(iter(dataloader))
        # check that the batch size is correct
        self.assertEqual(batch.num_graphs, batch_size, "The batch size is incorrect")
        # check that the number of nodes is correct
        self.assertLessEqual(batch.num_nodes, 1175, "The number of nodes is incorrect")
        # check that the number of edges is correct
        self.assertLessEqual(batch.num_edges, 10647, "The number of edges is incorrect")