Use Cases in GraphMuse
======================

This tutorial highlights common end-to-end workflows, from supervised training to evaluation and
model inference.

Supervised Training
-------------------

The following snippet demonstrates how to train a :class:`graphmuse.nn.MetricalGNN` on synthetic
score graphs produced with :func:`graphmuse.utils.create_random_music_graph`.

.. code-block:: python

   import numpy as np
   import torch
   from torch.optim import Adam
   from torch.nn import CrossEntropyLoss

   import graphmuse.nn as gmnn
   from graphmuse.loader import MuseNeighborLoader
   from graphmuse.utils import create_random_music_graph

   num_graphs = 10
   feature_size = 10
   num_classes = 4

   graphs = []
   for _ in range(num_graphs):
       graph = create_random_music_graph(
           graph_size=np.random.randint(100, 200),
           min_duration=1,
           max_duration=20,
           feature_size=feature_size,
           add_beat_nodes=True,
       )
       labels = np.random.randint(0, num_classes, graph["note"].x.shape[0])
       graph["note"].y = torch.tensor(labels, dtype=torch.long)
       graphs.append(graph)

   loader = MuseNeighborLoader(
       graphs,
       subgraph_size=50,
       batch_size=4,
       num_neighbors=[3, 3, 3],
   )

   metadata = graphs[0].metadata()
   model = gmnn.MetricalGNN(
       input_channels=feature_size,
       hidden_channels=32,
       output_channels=num_classes,
       num_layers=2,
       metadata=metadata,
   )

   optimizer = Adam(model.parameters(), lr=1e-3)
   criterion = CrossEntropyLoss()

   for epoch in range(5):
       model.train()
       total_loss = 0.0
       for batch in loader:
           optimizer.zero_grad()
           logits = model(batch.x_dict, batch.edge_index_dict)
           loss = criterion(logits, batch["note"].y)
           loss.backward()
           optimizer.step()
           total_loss += loss.item()

       print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader):.4f}")

Evaluation
----------

Leverage :mod:`sklearn.metrics` to evaluate predictions on held-out graphs.

.. code-block:: python

   from sklearn.metrics import accuracy_score

   model.eval()
   predictions, references = [], []
   with torch.no_grad():
       for batch in loader:
           logits = model(batch.x_dict, batch.edge_index_dict)
           preds = torch.argmax(logits, dim=1)
           predictions.extend(preds.cpu().numpy())
           references.extend(batch["note"].y.cpu().numpy())

   accuracy = accuracy_score(references, predictions)
   print(f"Accuracy: {accuracy:.3f}")

Model Inference
---------------

GraphMuse includes multiple architectures (``MetricalGNN``, ``CadenceGNN``, ``HybridGNN``) that share
a common interface. This example forwards synthetic data through the cadence detection model.

.. code-block:: python

   import torch

   cadence_gnn = gmnn.CadenceGNN(
       metadata=metadata,
       input_channels=feature_size,
       hidden_channels=32,
       output_channels=2,
       num_layers=3,
   )

   dummy_nodes = {"note": torch.rand((5, feature_size))}
   dummy_edges = {("note", "onset", "note"): torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])}
   logits = cadence_gnn(dummy_nodes, dummy_edges)
   print(logits.shape)

Where to go next
----------------

- Consult :mod:`graphmuse.nn` in the :ref:`api_reference` for the full list of neural modules.
- Combine the models with downstream evaluation datasets as described in the `GraphMuse paper
  <https://arxiv.org/abs/2407.12671>`_.
