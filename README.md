<p align="center">
   <img src="assets/graphmuse_long.png" height="300">
</p>

[![Latest Release](https://img.shields.io/github/v/release/manoskary/graphmuse)](https://github.com/manoskary/graphmuse/releases)
[![Pypi Package](https://badge.fury.io/py/graphmuse.svg)](https://badge.fury.io/py/graphmuse)
[![Unittest Status](https://github.com/manoskary/graphmuse/workflows/Tests/badge.svg)](https://github.com/manoskary/graphmuse/actions?query=workflow%3ATests)

# GraphMuse
GraphMuse is a Python Library for Graph Deep Learning on Symbolic Music.
This library intents to address Graph Deep Learning techniques and models applied specifically to Music Scores.

It contains a core set of graph-based music representations, based on Pytorch Geometric Data and HeteroData classes.
It includes functionalities for these graphs such as Sampling and several Graph Convolutional Networks.

The main core of the library includes sampling strategies for Music Score Graphs, Dataloaders, Graph Creation classes, and Graph Convolutional Networks.
The graph creation is implemented partly in C and works in unison with the Partitura library for parsing symbolic music.


It implements a variety of graph neural networks for music, including MusGConv, NoteGNN, MeasureGNn, BeatGNN, MetricalGNN, and HybridGNN.

Read the GraphMuse paper [here](https://arxiv.org/abs/2407.12671).

### Why GraphMuse?

GraphMuse is a library for symbolic music graph processing. It provides a set of tools for creating, manipulating, and learning from symbolic music graphs. It is built on top of PyTorch Geometric and provides a set of graph convolutional networks tailored to music data. GraphMuse is designed to be easy to use and flexible, allowing users to experiment with different graph representations and models for their music data.

GraphMuse aims to provide a set of tools for symbolic music graph processing that are easy to use and flexible. 


## Installation


GraphMuse is built on top of PyTorch and Pytorch Geometric. Therefore you need to install Pytorch and Pytorch Geometric first.

#### Pytorch and Pytorch Geometric Installation

First you need to install the Pytorch version suitable for your system.
You can find the instructions [here](https://pytorch.org/get-started/locally/).

You also need to install Pytorch Geometric. You can find the instructions [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
We recommend to use conda:
```shell
conda install pyg -c pyg
```

#### GraphMuse Installation

##### Using pip

You can install graphmuse along with the dependencies using pip:
```shell
pip install graphmuse
```

##### From Source

You can also install GraphMuse from source. First, clone the repository:
```shell
git clone https://github.com/manoskary/graphmuse.git
cd graphmuse
```

Then use pip for the rest of the dependencies:
```shell
pip install --verbose git+https://github.com/pyg-team/pyg-lib.git
pip install --verbose torch_scatter
pip install --verbose torch_sparse
pip install --verbose torch_cluster
pip install partitura
```

and install using the setup file:
```shell
python setup.py install
```

## Usage

### Graph Convolution

GraphMuse includes a variety of graph convolutional layers for music graphs.
Using the `MetricalGNN` model a simple example of a forward pass is shown below.


```python
import graphmuse.nn as gmnn
import torch

# Define the number of input features, output features, and edge features
num_input_features = 10
num_hidden_features = 10
num_output_features = 10
# metadata needs to be provided for the metrical graph similarly to Pytorch Geometric heterogeneous graph modules.
metadata = (
    ['note'],
    [('note', 'onset', 'note')]
)

# Create an instance of the MetricalGNN class
metrical_gnn = gmnn.MetricalGNN(num_input_features, num_hidden_features, num_output_features, metadata=metadata)

# Create some dummy data for the forward pass
num_nodes = 5
x_dict = {'note': torch.rand((num_nodes, num_input_features))}
edge_index_dict = {('note', 'onset', 'note'): torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])}

# Perform a forward pass
out = metrical_gnn(x_dict, edge_index_dict)

print(out)
```


### Score Graphs

GraphMuse includes accelerated implementations of graph creation for music scores.
You can create a score graph from a musicxml file using the [Partitura Python Library](https://github.com/CPJKU/partitura) and the `create_score_graph` function from GraphMuse.

```python
import graphmuse as gm
import partitura
import torch

score = partitura.load_musicxml('path_to_musicxml')
note_array = score.note_array()
feature_array = torch.rand((len(note_array), 10)) 
score_graph = gm.create_score_graph(feature_array, note_array)
print(score_graph)
```

### Sampling and Batching

GraphMuse includes a dataloader for sampling and batching music graphs together.
It uses the node-wise sampling strategy for each graph and batching them together.
You can specify the number of graphs to sample (`batch_size`) and the size of the subgraph to sample (`subgraph_size`).

Then, you can create an instance of the MuseNeighborLoader class by passing the list of graphs, the subgraph size, the batch size, and the number of neighbors as arguments.  

Finally, you can iterate over the dataloader to get batches of subgraphs.

```python
from graphmuse.loader import MuseNeighborLoader
from graphmuse.utils import create_random_music_graph
import numpy as np
import torch

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

# Create dataloader
dataloader = MuseNeighborLoader(graphs, subgraph_size=subgraph_size, batch_size=batch_size,
                                num_neighbors=[3, 3, 3])

# Iterate over the dataloader
for batch in dataloader:
    print(batch)
```


## Citing GraphMuse

GraphMuse was published at ISMIR 2024. To cite our work:
```bibtex
@inproceedings{karystinaios2024graphmuse,
  title={GraphMuse: A Library for Symbolic Music Graph Processing},
  author={Karystinaios, Emmanouil and Widmer, Gerhard},
  booktitle={Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)},
  year={2024}
}
```
