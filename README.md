# GraphMuse
A Graph Deep Learning Library for Music.

This library intents to address Graph Deep Learning techniques and models applied specifically on Music Scores.

### Usage

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

