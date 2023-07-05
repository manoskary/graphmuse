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

GraphMuse is built on top of PyTorch. Some additional dependencies are required to run the code:
- PyTorch Sparse
- PyTorch Scatter


## Installation

```shell
pip install graphmuse
```

To install using the setup file do:
```shell
python setup.py build_ext -i
```

## Usage

**Convolution**
```python
import graphmuse.nn as gmnn
import torch

conv = gmnn.SageConv(10, 10)
adj = torch.Tensor(
	[[0, 1, 0],
	 [1, 0, 1],
	 [0, 0, 1]])
x = torch.rand((3, 10))
h = conv(adj, x)
```

