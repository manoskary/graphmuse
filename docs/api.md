# API Documentation

## `graphmuse.io`

### `load_data_from_url(url)`
Load MIDI data from a URL.

**Parameters:**
- `url` (str): The URL to load the MIDI data from.

**Returns:**
- `MidiFile`: The loaded MIDI file.

**Example:**
```python
from graphmuse.io import load_data_from_url

url = "https://example.com/midi_file.mid"
midi_file = load_data_from_url(url)
```

### `is_url(input)`
Check if the input is a valid URL.

**Parameters:**
- `input` (str): The input string to check.

**Returns:**
- `bool`: True if the input is a valid URL, False otherwise.

**Example:**
```python
from graphmuse.io import is_url

input = "https://example.com"
is_valid = is_url(input)
```

### `load_midi_to_graph(path)`
Load a MIDI file and convert it to a score graph.

**Parameters:**
- `path` (str): The path to the MIDI file.

**Returns:**
- `graph`: The score graph.

**Example:**
```python
from graphmuse.io import load_midi_to_graph

path = "path/to/midi_file.mid"
score_graph = load_midi_to_graph(path)
```

## `graphmuse.loader`

### `MuseNeighborLoader`
Dataloader for MuseData objects. It samples a random region of a given budget from the graph.

**Parameters:**
- `graphs` (list): The list of graphs or an InMemoryDataset.
- `num_neighbors` (list or dict): The number of neighbors to sample for each node type.
- `subgraph_size` (int, optional): The size of the subgraph to sample, by default 100.
- `subgraph_sample_ratio` (float, optional): The ratio of subgraph samples to the total number of nodes, by default 2.
- `transform` (callable, optional): A function/transform that takes in a sampled subgraph and returns a transformed version, by default None.
- `transform_sampler_output` (callable, optional): A function/transform that takes in the output of the sampler and returns a transformed version, by default None.
- `filter_per_worker` (bool, optional): Whether to filter the data per worker, by default None.
- `custom_cls` (HeteroData, optional): A custom HeteroData class, by default None.
- `device` (str or torch.device, optional): The device to use, by default "cpu".
- `is_sorted` (bool, optional): Whether the data is sorted, by default False.
- `share_memory` (bool, optional): Whether to share memory, by default False.
- `order_batch` (bool, optional): Whether to order the batch, by default True.

**Example:**
```python
from graphmuse.loader import MuseNeighborLoader
from torch_geometric.data import HeteroData

data = HeteroData()
data['note'].x = torch.randn(100, 16)
data['note', 'consecutive', 'note'].edge_index = torch.randint(0, 100, (2, 500))
loader = MuseNeighborLoader([data], num_neighbors=[10, 5], subgraph_size=50)

for batch in loader:
    print(batch)
```

## `graphmuse.nn`

### `GraphAttentionLayer`
Graph Attention Layer (GAT) implementation.

**Parameters:**
- `input_channels` (int): Number of input channels.
- `output_channels` (int): Number of output channels.
- `dropout` (float): Dropout rate.
- `alpha` (float): Negative slope for LeakyReLU.
- `concat` (bool, optional): Whether to concatenate the output of multiple heads, by default True.

**Example:**
```python
from graphmuse.nn import GraphAttentionLayer
import torch

gat_layer = GraphAttentionLayer(input_channels=16, output_channels=8, dropout=0.5, alpha=0.2)
h = torch.randn(10, 16)
adj = torch.randint(0, 2, (10, 10))
out = gat_layer(h, adj)
print(out.shape)
```

### `CustomGATConv`
The graph attentional operator from the "Graph Attention Networks" paper.

**Parameters:**
- `in_channels` (int or tuple): Size of each input sample, or -1 to derive the size from the first input(s) to the forward method.
- `out_channels` (int): Size of each output sample.
- `heads` (int, optional): Number of multi-head-attentions, by default 1.
- `concat` (bool, optional): If set to False, the multi-head attentions are averaged instead of concatenated, by default True.
- `negative_slope` (float, optional): LeakyReLU angle of the negative slope, by default 0.2.
- `dropout` (float, optional): Dropout probability of the normalized attention coefficients, by default 0.
- `add_self_loops` (bool, optional): If set to False, will not add self-loops to the input graph, by default True.
- `edge_dim` (int, optional): Edge feature dimensionality, by default None.
- `fill_value` (float or torch.Tensor or str, optional): The way to generate edge features of self-loops, by default "mean".
- `norm_msg` (bool, optional): Whether to normalize the message, by default True.
- `bias` (bool, optional): If set to False, the layer will not learn an additive bias, by default True.

**Example:**
```python
from graphmuse.nn import CustomGATConv
import torch

in_channels = 16
out_channels = 8
heads = 3
gat_conv = CustomGATConv(in_channels, out_channels, heads)
x = torch.randn(10, 16)
edge_index = torch.randint(0, 10, (2, 20))
out = gat_conv(x, edge_index)
print(out.shape)
```

### `CadenceGNN`
CadenceGNN is a graph neural network model for cadence detection in music.

**Parameters:**
- `metadata` (dict): Metadata for the graph.
- `input_channels` (int): Number of input channels.
- `hidden_channels` (int): Number of hidden channels.
- `output_channels` (int): Number of output channels.
- `num_layers` (int): Number of layers in the GNN.
- `dropout` (float, optional): Dropout rate, by default 0.5.
- `hybrid` (bool, optional): Whether to use a hybrid model with RNN, by default False.

**Example:**
```python
from graphmuse.nn import CadenceGNN
import torch

metadata = {"note": {"x": torch.randn(10, 16)}}
model = CadenceGNN(metadata, input_channels=16, hidden_channels=32, output_channels=2, num_layers=3)
x_dict = {"note": torch.randn(10, 16)}
edge_index_dict = {"note": {"note": torch.randint(0, 10, (2, 20))}}
out = model(x_dict, edge_index_dict)
print(out.shape)
```

## `graphmuse.utils`

### `get_pc_one_hot(note_array)`
Get one-hot encoding for pitch classes.

**Parameters:**
- `note_array` (np.ndarray): The note array.

**Returns:**
- `np.ndarray`: The one-hot encoded pitch classes.
- `list`: The names of the pitch classes.

**Example:**
```python
from graphmuse.utils import get_pc_one_hot
import numpy as np

note_array = np.array([{"pitch": 60}, {"pitch": 62}, {"pitch": 64}])
one_hot, names = get_pc_one_hot(note_array)
print(one_hot)
print(names)
```

### `get_octave_one_hot(note_array)`
Get one-hot encoding for octaves.

**Parameters:**
- `note_array` (np.ndarray): The note array.

**Returns:**
- `np.ndarray`: The one-hot encoded octaves.
- `list`: The names of the octaves.

**Example:**
```python
from graphmuse.utils import get_octave_one_hot
import numpy as np

note_array = np.array([{"pitch": 60}, {"pitch": 62}, {"pitch": 64}])
one_hot, names = get_octave_one_hot(note_array)
print(one_hot)
print(names)
```

### `get_score_features(score)`
Returns features Voice Detection features.

**Parameters:**
- `score` (partitura.Score): The score.

**Returns:**
- `np.ndarray`: The features.
- `list`: The names of the features.

**Example:**
```python
from graphmuse.utils import get_score_features
import partitura

score = partitura.load_musicxml('path_to_musicxml')
features, names = get_score_features(score)
print(features)
print(names)
```

### `MapDict`
A dictionary that allows attribute-style access.

**Example:**
```python
from graphmuse.utils import MapDict

d = MapDict(a=1, b=2)
print(d.a)
d.c = 3
print(d.c)
```
