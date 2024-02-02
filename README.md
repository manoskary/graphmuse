# GraphMuse
GraphMuse is a Graph Deep Learning Library for Music.

This library intents to address Graph Deep Learning techniques and models applied specifically on Music Scores.

It contains a core set of graph-based music representations, such as a Heterogeneous and a Homogeneous Score Graph class.
It includes functionalities for this graphs such as saving, loading and batching graphs together.

The main core of the library includes accelarated SOTA sampling strategies for Large Graphs, 
which are implemented in C11 and CUDA. 


It implements a variety of graph neural networks for music, including Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), GraphSAGE, and Graph Isomorphism Networks (GIN).
It also implements a variety of graph neural networks for music, including Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), GraphSAGE, and Graph Isomorphism Networks (GIN).
Modules of the library contain implementations of the following models:
- Contrastive Graph Neural Networks similar to SimCLR;
- Hierarchical Graph Auto-Encoders with edge Polling;
- Hyperbolic Graph Neural Networks with Poincare Topology.

### Dependencies

GraphMuse is built on top of PyTorch and Pytorch Geometric. Some additional dependencies are required to run the code:
- PyTorch Sparse
- PyTorch Scatter


## Installation

To install Graphmuse you first need to install the Pytorch version suitable for your system.
You can find the instructions [here](https://pytorch.org/get-started/locally/).

You also need to install Pytorch Geometric. You can find the instructions [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
We recommend to use conda:
```shell
conda install conda install pyg -c pyg
```

You can install graphmuse along with the dependencies using pip:
```shell
pip install graphmuse
```



Or use pip for the rest of the dependencies:
```shell
pip install --verbose git+https://github.com/pyg-team/pyg-lib.git
pip install --verbose torch_scatter
pip install --verbose torch_sparse
pip install --verbose torch_cluster
pip install partitura
```

and install using the setup file:
```shell
python setup.py build_ext -i
```

## Usage

### Convolution
```python
import graphmuse.nn as gmnn
import torch

conv = gmnn.MusGConv(10, 10, 10)
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.rand((3, 10))
edge_features = torch.rand((4, 10))
h = conv(x, edge_index, edge_features)
```


### Score Graphs

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

GraphMuse includes a dataloader for sampling and batching graphs together.
It uses the node-wise sampling strategy for each graph and batching them together.
You can specify the number of graphs to sample (`batch_size`) and the size of the subgraph to sample (`subgraph_size`).

```python
import graphmuse as gm

scores_graphs = ["list of score graphs"]
dataloader = gm.loader.MusGraphDataLoader(scores_graphs, num_neighbors=[3, 3], batch_size=32, subgraph_size=100)
print(next(iter(dataloader)))
```
