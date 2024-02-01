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
- Hierachical Graph Auto-Encoders with edge Polling;
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

**Convolution**
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

