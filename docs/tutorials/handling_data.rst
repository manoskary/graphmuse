Handling Data in GraphMuse
==========================

This tutorial walks through the data utilities bundled with GraphMuse, from constructing score
graphs to sampling sub-graphs suitable for model training.

Creating Score Graphs
---------------------

GraphMuse interoperates with `Partitura <https://github.com/CPJKU/partitura>`_ to parse MusicXML and
convert the resulting note arrays into heterogeneous score graphs.

.. code-block:: python

   import partitura
   import torch

   import graphmuse as gm

   score = partitura.load_musicxml("path/to/score.musicxml")
   note_array = score.note_array()

   # Random features for demonstration purposes
   features = torch.rand((len(note_array), 12))

   score_graph = gm.create_score_graph(features, note_array)
   print(score_graph)

Each score graph is a :class:`torch_geometric.data.HeteroData` instance containing node features for
notes, optional beat nodes, and the relevant edge relationships.

Sampling and Batching
---------------------

The :class:`graphmuse.loader.MuseNeighborLoader` wraps PyTorch Geometric's neighbor sampling API with
music-aware defaults.

.. code-block:: python

   from graphmuse.loader import MuseNeighborLoader

   loader = MuseNeighborLoader(
       [score_graph],
       subgraph_size=64,
       batch_size=2,
       num_neighbors=[3, 3, 3],
   )

   for batch in loader:
       print(batch)

The loader yields mini-batches that preserve the heterogeneous structure of the underlying graph,
ready for consumption by the neural network modules in :mod:`graphmuse.nn`.

Working with Synthetic Graphs
-----------------------------

The :mod:`graphmuse.utils` module includes helpers for generating random graphs that mimic the
structure of symbolic music. These are designed for unit tests and experimentation when real data
is unavailable.

.. code-block:: python

   import numpy as np
   import torch

   from graphmuse.loader import MuseNeighborLoader
   from graphmuse.utils import create_random_music_graph

   graphs = []
   for _ in range(5):
       graph = create_random_music_graph(
           graph_size=np.random.randint(100, 200),
           min_duration=1,
           max_duration=20,
           feature_size=10,
           add_beat_nodes=True,
       )
       # Optionally attach labels for supervised tasks
       labels = np.random.randint(0, 4, graph["note"].x.shape[0])
       graph["note"].y = torch.tensor(labels, dtype=torch.long)
       graphs.append(graph)

   loader = MuseNeighborLoader(graphs, subgraph_size=50, batch_size=4, num_neighbors=[3, 3, 3])
   first_batch = next(iter(loader))
   print(first_batch["note"].x.shape)

Next Steps
----------

Continue with :doc:`use_cases` to train and evaluate GraphMuse models on the sampled data.
