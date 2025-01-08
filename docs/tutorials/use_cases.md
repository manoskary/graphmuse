# Use Cases in GraphMuse

This tutorial will guide you through specific use cases in GraphMuse, including training a model on a dataset and evaluating its performance.

## Training a Model on a Dataset

GraphMuse provides tools to train models on music graph datasets. In this example, we will demonstrate how to train a `MetricalGNN` model on a dataset of music graphs.

### Example: Training a MetricalGNN Model

```python
import graphmuse.nn as gmnn
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from graphmuse.loader import MuseNeighborLoader
from graphmuse.utils import create_random_music_graph

# Create a random dataset of music graphs
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

# Define the model, optimizer, and loss function
num_input_features = 10
num_hidden_features = 10
num_output_features = 4
num_layers = 1
metadata = (
    ['note'],
    [('note', 'onset', 'note')]
)
model = gmnn.MetricalGNN(num_input_features, num_hidden_features, num_output_features, num_layers, metadata=metadata)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}')
```

## Evaluating the Model's Performance

After training the model, we need to evaluate its performance on a test dataset. In this example, we will demonstrate how to evaluate the `MetricalGNN` model's performance.

### Example: Evaluating the MetricalGNN Model

```python
import torch
from sklearn.metrics import accuracy_score

# Create a test dataset of music graphs
num_test_graphs = 5
test_graphs = list()
for i in range(num_test_graphs):
    l = np.random.randint(min_nodes, max_nodes)
    graph = create_random_music_graph(
        graph_size=l, min_duration=min_dur, max_duration=max_dur, feature_size=feature_size, add_beat_nodes=True)
    label = np.random.randint(0, labels, graph["note"].x.shape[0])
    graph["note"].y = torch.tensor(label, dtype=torch.long)
    test_graphs.append(graph)

# Create test dataloader
test_dataloader = MuseNeighborLoader(test_graphs, subgraph_size=subgraph_size, batch_size=batch_size,
                                     num_neighbors=[3, 3, 3])

# Evaluation loop
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_dataloader:
        out = model(batch.x_dict, batch.edge_index_dict)
        preds = torch.argmax(out, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f'Test Accuracy: {accuracy}')
```

## Conclusion

In this tutorial, we have covered specific use cases in GraphMuse, including training a model on a dataset and evaluating its performance. By following these steps, you can effectively train and evaluate models on symbolic music data using GraphMuse.
