from unittest import TestCase
import graphmuse.nn as gmnn
import torch
import graphmuse as gm
import partitura
from graphmuse.loader import MuseNeighborLoader
from graphmuse.utils import create_random_music_graph
import numpy as np

class TestReadMeCode(TestCase):
    def test_metricalgnn(self):
        # Define the number of input features, output features, and edge features
        num_input_features = 10
        num_hidden_features = 10
        num_output_features = 10
        num_layers = 1
        # metadata needs to be provided for the metrical graph similarly to Pytorch Geometric heterogeneous graph modules.
        metadata = (
            ['note'],
            [('note', 'onset', 'note')]
        )

        # Create an instance of the MetricalGNN class
        metrical_gnn = gmnn.MetricalGNN(num_input_features, num_hidden_features, num_output_features, num_layers,
                                        metadata=metadata)

        # Create some dummy data for the forward pass
        num_nodes = 5
        x_dict = {'note': torch.rand((num_nodes, num_input_features))}
        edge_index_dict = {('note', 'onset', 'note'): torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])}

        # Perform a forward pass
        out = metrical_gnn(x_dict, edge_index_dict)

        print(out)
        self.assertEqual(out.shape, (num_nodes, num_output_features))

    def test_score_graph(self):
        score = partitura.load_musicxml(partitura.EXAMPLE_MUSICXML)
        note_array = score.note_array()
        feature_array = torch.rand((len(note_array), 10))
        score_graph = gm.create_score_graph(feature_array, note_array)
        print(score_graph)
        self.assertTrue(True)

    def test_muse_neighbor_loader(self):

        # Create a random graph
        num_graphs = 10
        max_nodes = 200
        min_nodes = 100
        max_dur = 20
        min_dur = 1
        subgraph_size = 50
        batch_size = 4
        feature_size = 10
        labels = 4

        graphs = list()
        for i in range(num_graphs):
            l = np.random.randint(min_nodes, max_nodes)
            graph = create_random_music_graph(
                graph_size=l, min_duration=min_dur, max_duration=max_dur, feature_size=feature_size,
                add_beat_nodes=True)
            label = np.random.randint(0, labels, graph["note"].x.shape[0])
            graph["note"].y = torch.tensor(label, dtype=torch.long)
            graphs.append(graph)

        # Create dataloader
        dataloader = MuseNeighborLoader(graphs, subgraph_size=subgraph_size, batch_size=batch_size,
                                        num_neighbors=[3, 3, 3])

        # Iterate over the dataloader
        for batch in dataloader:
            print(batch)
            break

        self.assertTrue(True)