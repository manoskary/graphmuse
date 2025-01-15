Handling Data in GraphMuse
==========================

This tutorial will guide you through the process of handling data in GraphMuse, including creating score graphs and using the provided models with examples.

Creating Score Graphs
----------------------

GraphMuse provides a convenient way to create score graphs from symbolic music data. The `create_score_graph` function allows you to create a graph representation of a music score.

Example: Creating a Score Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import graphmuse as gm
    import partitura
    import torch

    # Load a music score using Partitura
    score = partitura.load_musicxml('path_to_musicxml')
    note_array = score.note_array()

    # Generate random features for the notes
    feature_array = torch.rand((len(note_array), 10))

    # Create a score graph
    score_graph = gm.create_score_graph(feature_array, note_array)
    print(score_graph)

In this example, we load a music score using Partitura, generate random features for the notes, and create a score graph using the `create_score_graph` function.

Using the Provided Models
--------------------------

GraphMuse includes several pre-defined models for processing music graphs. In this section, we will demonstrate how to use these models with examples.

Example: Using the MetricalGNN Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `MetricalGNN` model is a graph neural network designed for processing music graphs. Here is an example of how to use it:

.. code-block:: python

    import graphmuse.nn as gmnn
    import torch

    # Define the number of input features, output features, and edge features
    num_input_features = 10
    num_hidden_features = 10
    num_output_features = 10
    num_layers = 1

    # Metadata needs to be provided for the metrical graph similarly to Pytorch Geometric heterogeneous graph modules.
    metadata = (
        ['note'],
        [('note', 'onset', 'note')]
    )

    # Create an instance of the MetricalGNN class
    metrical_gnn = gmnn.MetricalGNN(num_input_features, num_hidden_features, num_output_features, num_layers, metadata=metadata)

    # Create some dummy data for the forward pass
    num_nodes = 5
    x_dict = {'note': torch.rand((num_nodes, num_input_features))}
    edge_index_dict = {('note', 'onset', 'note'): torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])}

    # Perform a forward pass
    out = metrical_gnn(x_dict, edge_index_dict)
    print(out)

In this example, we define the number of input features, output features, and edge features, create an instance of the `MetricalGNN` class, and perform a forward pass with some dummy data.

Data Handling Pipeline
----------------------

The data handling pipeline in GraphMuse involves several steps, including data loading, graph creation, sampling, and model training. Here is an overview of the pipeline:

1. **Data Loading**: Load symbolic music data using libraries like Partitura.
2. **Graph Creation**: Create score graphs from the loaded data using the `create_score_graph` function.
3. **Sampling**: Use the provided sampling strategies to sample subgraphs from the score graphs.
4. **Data Loading**: Utilize the data loaders to batch and load the sampled subgraphs.
5. **Model Training**: Train the provided neural network models on the batched subgraphs.
6. **Evaluation**: Evaluate the trained models on test data to assess their performance.

Example: Data Handling Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import graphmuse as gm
    import partitura
    import torch
    from graphmuse.loader import MuseNeighborLoader

    # Load a music score using Partitura
    score = partitura.load_musicxml('path_to_musicxml')
    note_array = score.note_array()

    # Generate random features for the notes
    feature_array = torch.rand((len(note_array), 10))

    # Create a score graph
    score_graph = gm.create_score_graph(feature_array, note_array)

    # Create a list of score graphs (for demonstration purposes)
    score_graphs = [score_graph]

    # Create a data loader for sampling and batching
    dataloader = MuseNeighborLoader(score_graphs, subgraph_size=50, batch_size=4, num_neighbors=[3, 3, 3])

    # Iterate over the data loader to get batches of subgraphs
    for batch in dataloader:
        print(batch)

In this example, we demonstrate the data handling pipeline by loading a music score, creating a score graph, and using the `MuseNeighborLoader` to sample and batch subgraphs.

Conclusion
----------

In this tutorial, we have covered how to handle data in GraphMuse, including creating score graphs, using the provided models, and understanding the data handling pipeline. By following these steps, you can effectively process and analyze symbolic music data using GraphMuse.
