.. STEP documentation master file, created by
   sphinx-quickstart on Thu Jan 11 17:09:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to STEP's documentation!
================================

.. image:: http://docs.3s540lab.cloud/images/STEP_fig_1a.webp
   :alt: 
   :align: center

Overview
--------
STEP, an acronym for Spatial Transcriptomics Embedding Procedure, 
is a foundation deep learning/AI architecture for the analysis of spatially resolved transcriptomics (SRT) data, 
and is also compatible with scRNA-seq data. 
STEP roots on the precise captures of three major varitions occured in the SRT (and scRNA-seq) data: 
Transcriptional Variations, Batch Variations and Spatial Variations with the correponding modular designs: 
Backbone model: a Transformer based model togther with gene module seqeunce mapping; 
Batch-effect model: A pair of inverse transformations utilizing the batch-embedding conception for the decoupled batch-effect elimination; 
Spatial model: a GCN-based spatial filter/smoother working on the extracted embedding from the Backbone model, different from the usage of GCN in other methods as a feature extractor. 
Thus, with the proper combinations of these models, STEP introduces a unified approach to systematically process and analyze single or multiple samples of SRT data, disregarding location relationships between sections (meaning both contiguous and non-contiguous sections), to reveal multi-scale bilogical heterogeneities (cell types and spatial domains) in multi-resolution SRT data. 
Furthermore, STEP can also conduct integrative analysis on scRNA-seq and SRT data.

Contents:
---------
.. toctree::
   :maxdepth: 3
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
