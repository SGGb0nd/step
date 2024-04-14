.. STEP documentation master file, created by
   sphinx-quickstart on Thu Jan 11 17:09:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to STEP's documentation!
================================

STEP, an acronym for Spatial Transcriptomics Embedding Procedure, 
is a deep learning-based tool for the analysis of single-cell RNA (scRNA-seq) and spatially resolved transcriptomics (SRT) data. 
step introduces a unified approach to process and analyze multiple samples of scRNA-seq data as well as align several sections of SRT data, disregarding location relationships. 
Furthermore, step conducts integrative analysis across different modalities like scRNA-seq and SRT.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

.. automodule:: step
   :members:

.. nbgallery::
   :maxdepth: 1
   :caption: Tutorials:

   notebooks/DLFPC
   notebooks/Human_Lymph_Node
   notebooks/MERFISH
   notebooks/scRNA-seq

.. toctree::
   :maxdepth: 2
   :caption: API:

   step
   step.functionality
   step.manager
   step.models
   step.modules
   step.utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
