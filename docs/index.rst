.. _graphmuse:

GraphMuse Documentation
=======================

GraphMuse is a research toolkit for graph-based deep learning on symbolic music. It provides
heterogeneous graph representations for scores, sampling utilities, and graph neural network
architectures that are tailored to music analysis tasks.

Quick Links
-----------

- :doc:`Getting started <getting_started>`
- :doc:`Handling data <tutorials/handling_data>`
- :doc:`Use cases <tutorials/use_cases>`
- :ref:`api_reference`

Features
--------

- Score graph creation pipelines with accelerated C extensions for performant preprocessing.
- Sampling strategies and dataloaders optimised for symbolic music graphs.
- A collection of graph neural network layers and models designed for metrical, cadence, and hybrid tasks.
- Utilities to generate synthetic graphs for experimentation and prototyping.

Repository Structure
--------------------

- ``graphmuse`` – Library source code.
  - ``io`` – Input/output utilities for interacting with score files.
  - ``loader`` – Sampling dataloaders for heterogeneous music graphs.
  - ``nn`` – Graph neural network modules and reference models.
  - ``samplers`` – Interfaces to high-performance C sampling routines.
  - ``utils`` – Helpers for graph creation and experimentation.
- ``include`` – C headers that back the sampling extension.
- ``docs`` – Documentation sources prepared for Read the Docs.
- ``tests`` – PyTest-based regression tests.

Processing Pipeline
-------------------

The typical workflow when using GraphMuse is:

1. Create a score graph from symbolic music (e.g., MusicXML) via :func:`graphmuse.create_score_graph`.
2. Sample sub-graphs with :class:`graphmuse.loader.MuseNeighborLoader` or other strategies.
3. Train music-specific graph neural networks from :mod:`graphmuse.nn`.
4. Evaluate the resulting model on held-out data.

.. figure:: https://raw.githubusercontent.com/manoskary/graphmuse/main/assets/graphmuse_pipeline.png
   :height: 150
   :align: center

   The high-level GraphMuse pipeline.

.. _api_reference:

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started
   tutorials/handling_data
   tutorials/use_cases
   api

