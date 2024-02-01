from graphmuse.loader import MuseNeighborLoader
import numpy as np
from graphmuse.samplers import c_set_seed
from graphmuse.utils import create_random_music_graph
import torch
from unittest import TestCase
import torch.nn as nn
from torch_geometric.nn import SAGEConv, to_hetero


# Standardize the random seed
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
c_set_seed(42)
torch.backends.cudnn.deterministic = True


class GNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, dropout=0.5):
        super(GNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, output_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(output_dim, output_dim))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.normalize = nn.LayerNorm(output_dim)

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.normalize(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return x


class TestMuseNeighborLoader(TestCase):
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

        graphs = list()
        for i in range(num_graphs):
            l = np.random.randint(min_nodes, max_nodes)
            graph = create_random_music_graph(
                graph_size=l, min_duration=min_dur, max_duration=max_dur, feature_size=feature_size, add_beat_nodes=True)
            graphs.append(graph)

        # create dataloader
        dataloader = MuseNeighborLoader(graphs, subgraph_size=subgraph_size, batch_size=batch_size,
                                        num_neighbors=[3, 3])
        metadata = (["note", "beat"], [
            ("note", "onset", "note"),
            ("note", "consecutive", "note"),
            ("note", "rest", "note"),
            ("note", "during", "note"),
            ("note", "consecutive_rev", "note"),
            ("note", "rest_rev", "note"),
            ("note", "during_rev", "note"),
            ("note", "connects", "beat"),
            ("beat", "connects", "note"),
            ("beat", "next", "beat"),
        ])
        model = to_hetero(GNN(feature_size, 20, 2), metadata)
        batch = next(iter(dataloader))

        # input to the model
        out = model(batch.x_dict, batch.edge_index_dict)
        batch["note"].x = out["note"]
        target_outputs = torch.cat([data["note"].x[:subgraph_size] for data in batch.to_data_list()], dim=0)

        # check that the batch size is correct
        self.assertEqual(batch.num_graphs, batch_size, "The batch size is incorrect")
        # check that the number of nodes is correct
        self.assertLessEqual(batch.num_nodes, 1877, "The number of nodes is incorrect")
        # check that the number of edges is correct
        self.assertLessEqual(batch.num_edges, 13806, "The number of edges is incorrect")
        # check that the number of node types is correct
        self.assertEqual(len(batch.node_types), 2, "The number of node types is incorrect")
        # check that the number of edge types is correct
        self.assertEqual(len(batch.edge_types), 10, "The number of edge types is incorrect")
        # check that the output shape is correct for the target node type
        self.assertEqual(target_outputs.shape, (batch_size*subgraph_size, 20), "The output shape is incorrect")