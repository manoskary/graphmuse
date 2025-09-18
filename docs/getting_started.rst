Getting Started
===============

This guide covers the basic installation steps for GraphMuse and provides a minimal example to
verify your environment.

Installation
------------

1. Create a fresh Python environment (conda is recommended)::

       conda create -n graphmuse python=3.11
       conda activate graphmuse

2. Install PyTorch following the `official instructions <https://pytorch.org/get-started/locally/>`_.
3. Install the PyTorch Geometric stack as described in the
   `PyG installation notes <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_.
4. Install GraphMuse from PyPI::

       pip install graphmuse

   To work with the latest development version, clone the repository and install in editable mode::

       git clone https://github.com/manoskary/graphmuse.git
       cd graphmuse
       pip install -e .

Troubleshooting
---------------

- On some platforms the optional ``pyg-lib`` wheels may fail to build automatically. Install them
  manually with::

       pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

  Replace ``${TORCH}`` and ``${CUDA}`` with the versions that match your PyTorch installation. Use
  ``cpu`` if you do not require GPU support.
- If C++ build tools are missing for the ``torch-scatter`` or similar packages, reinstall them after
  configuring a compiler toolchain. Refer to the PyG docs linked above for platform-specific help.

Quick Start
-----------

Create a score graph from a MusicXML file and run a forward pass with a bundled model:

.. code-block:: python

   import numpy as np
   import torch
   import partitura

   import graphmuse as gm
   from graphmuse.loader import MuseNeighborLoader
   from graphmuse.nn import MetricalGNN

   # Load symbolic music with Partitura
   score = partitura.load_musicxml("path/to/score.musicxml")
   note_array = score.note_array()

   # Construct features and a score graph
   features = torch.rand((len(note_array), 16))
   score_graph = gm.create_score_graph(features, note_array)

   # Build a mini-batch sampler
   loader = MuseNeighborLoader(
       [score_graph],
       subgraph_size=32,
       batch_size=1,
       num_neighbors=[3, 3, 3],
   )

   # Define the model metadata (node types and edge types)
   metadata = score_graph.metadata()
   model = MetricalGNN(
       num_input_features=16,
       num_hidden_features=32,
       num_output_features=4,
       num_layers=2,
       metadata=metadata,
   )

   batch = next(iter(loader))
   logits = model(batch.x_dict, batch.edge_index_dict)
   print(logits.shape)

Next Steps
----------

- Explore :doc:`tutorials/handling_data` for more details on building pipelines.
- See :doc:`tutorials/use_cases` for training and evaluation patterns.
- Dive into the :ref:`api_reference` for the full module documentation.
