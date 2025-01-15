.. _graphmuse:

GraphMuse Documentation
=======================

Introduction
------------

GraphMuse is a Python Library for Graph Deep Learning on Symbolic Music. This library aims to address Graph Deep Learning techniques and models applied specifically to Music Scores. It contains a core set of graph-based music representations, based on Pytorch Geometric Data and HeteroData classes. It includes functionalities for these graphs such as Sampling and several Graph Convolutional Networks.

The main core of the library includes sampling strategies for Music Score Graphs, Dataloaders, Graph Creation classes, and Graph Convolutional Networks. The graph creation is implemented partly in C and works in unison with the Partitura library for parsing symbolic music.

Repository Structure
--------------------

The repository is structured as follows:

- `graphmuse`: Contains the main library code.
  - `io`: Input/output utilities.
  - `loader`: Data loaders and samplers.
  - `nn`: Neural network modules and models.
  - `samplers`: Sampling strategies for graphs.
  - `utils`: Utility functions and classes.
- `tests`: Contains test files to validate the functionality of the code.
- `include`: Contains C source files and headers for performance optimization.
- `docs`: Contains the documentation and tutorials.

Available Modules
-----------------

### `graphmuse`

- `io`: Provides utilities for input and output operations.
- `loader`: Contains data loaders and samplers for handling graph data.
- `nn`: Includes various neural network modules and models for graph-based learning.
- `samplers`: Implements different sampling strategies for graphs.
- `utils`: Offers utility functions and classes for graph manipulation and processing.

Processing Pipeline
-------------------

The GraphMuse processing pipeline involves the following steps:

1. **Graph Creation**: Create a score graph from a music score using the `create_score_graph` function.
2. **Sampling**: Use the provided sampling strategies to sample subgraphs from the score graph.
3. **Data Loading**: Utilize the data loaders to batch and load the sampled subgraphs.
4. **Model Training**: Train the provided neural network models on the batched subgraphs.
5. **Evaluation**: Evaluate the trained models on test data to assess their performance.

The pipeline is designed to be flexible and modular, allowing users to experiment with different graph representations, sampling strategies, and models for their music data.

.. image:: https://raw.githubusercontent.com/manoskary/graphmuse/main/assets/graphmuse_pipeline.png
   :height: 150
   :align: center

.. image:: https://raw.githubusercontent.com/manoskary/graphmuse/main/assets/satie_graph.png
   :height: 200
   :align: center

.. image:: https://raw.githubusercontent.com/manoskary/graphmuse/main/assets/sampling_graphmuse.png
   :height: 400
   :align: center
