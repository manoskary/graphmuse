from graphmuse.loader import MuseNeighborLoader, transform_to_pyg
import numpy as np
from graphmuse.samplers import c_set_seed
from graphmuse.utils import create_random_music_graph
import torch
from unittest import TestCase
import torch.nn as nn
import time
from torch_geometric.nn import SAGEConv, to_hetero
from graphmuse.nn.models.metrical_gnn import MetricalGNN


# Standardize the random seed
seed = 2
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
c_set_seed(seed)
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
        labels = 4

        graphs = list()
        for i in range(num_graphs):
            l = np.random.randint(min_nodes, max_nodes)
            graph = create_random_music_graph(
                graph_size=l, min_duration=min_dur, max_duration=max_dur, feature_size=feature_size, add_beat_nodes=True)
            label = np.random.randint(0, labels, graph["note"].x.shape[0])
            graph["note"].y = torch.tensor(label, dtype=torch.long)
            graphs.append(graph)

        metadata = graph.metadata()
        # create dataloader
        dataloader = MuseNeighborLoader(graphs, subgraph_size=subgraph_size, batch_size=batch_size,
                                        num_neighbors=[3, 3, 3])


        batch = next(iter(dataloader))

        # input to a model
        # model = to_hetero(GNN(feature_size, 20, 2), batch.metadata())


        model = MetricalGNN(feature_size, 64, labels, 3, metadata)
        loss = nn.CrossEntropyLoss()
        # out = model(batch.x_dict, batch.edge_index_dict)
        neighbor_mask_node = {k: batch[k].neighbor_mask for k in batch.node_types}
        neighbor_mask_edge = {k: batch[k].neighbor_mask for k in batch.edge_types}
        start = time.time()
        target_outputs = model(batch.x_dict, batch.edge_index_dict,
                               neighbor_mask_node, neighbor_mask_edge)
        mask_model_time = time.time() - start
        # Trim the labels to the target nodes (i.e. layer 0)
        target_labels = batch["note"].y[neighbor_mask_node["note"] == 0]
        loss = loss(target_outputs, target_labels)
        print("The loss is: ", loss.item())
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
        self.assertEqual(target_outputs.shape, (batch_size*subgraph_size, labels), "The output shape is incorrect")
        # check that transformation is correct
        x_before_transform = batch["note"].x[batch["note"].neighbor_mask == 0]
        batch_transform = transform_to_pyg(batch, dataloader.num_neighbors.num_hops)
        x_after_transform = batch_transform["note"].x[:batch_transform["note"].batch_size]
        self.assertTrue(torch.allclose(x_before_transform, x_after_transform),
                        "The x values are not equal after transformation")
        fast_model = MetricalGNN(feature_size, 64, labels, 3, metadata, fast=True)
        start = time.time()
        x_after_transform = fast_model(batch_transform.x_dict, batch_transform.edge_index_dict,
                   batch_transform.num_sampled_nodes_dict, batch_transform.num_sampled_edges_dict)
        x_after_transform = x_after_transform[:batch_transform["note"].batch_size]
        fast_model_time = time.time() - start

        self.assertEqual(x_after_transform.shape, (batch_size*subgraph_size, labels), "The output shape is incorrect")
        print("The time for the mask model is: ", mask_model_time)
        print("The time for the fast model is: ", fast_model_time)