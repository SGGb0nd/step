# STEP: Spatial Transcriptomics Embedding Procedure
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/SGGb0nd/step-dev/CI)](https://github.com/SGGb0nd/step-dev/actions)

![image](http://docs.3s540lab.cloud/images/STEP_fig_1a.webp)
![image](http://docs.3s540lab.cloud/images/STEP_fig_1b.webp)

## Introduction
STEP, an acronym for Spatial Transcriptomics Embedding Procedure, is a foundation deep learning/AI architecture for the analysis of spatially resolved transcriptomics (SRT) data, and is also compatible with scRNA-seq data. STEP roots on the precise captures of three major varitions occured in the SRT (and scRNA-seq) data: **Transcriptional Variations**, **Batch Variations** and **Spatial Variations** with the correponding modular designs: **Backbone model**: a Transformer based model togther with gene module seqeunce mapping; **Batch-effect model**: A pair of inverse transformations utilizing the *batch-embedding* conception for the decoupled batch-effect elimination; **Spatial model**: a GCN-based spatial filter/smoother working on the extracted embedding from the Backbone model, different from the usage of GCN in other methods as a feature extractor. Thus, with the proper combinations of these models, STEP introduces a unified approach to systematically process and analyze single or multiple samples of SRT data, disregarding location relationships between sections (meaning both contiguous and non-contiguous sections), to reveal multi-scale bilogical heterogeneities (cell types and spatial domains) in multi-resolution SRT data. Furthermore, STEP can also conduct integrative analysis on scRNA-seq and SRT data.

## Key Features

-  Integration of multiple scRNA-seq and single-cell resolution SRT samples to reveal cell-type level heterogeneities.
-  Alignment of various SRT data sections contiguous or non-contiguous to identify spatial domains across sections.
-  Scalable to different data resolutions, i.e., wild range of technologies and platforms of SRT data, including **Visium HD**, **Visum**, **MERFISH**, **STARmap**, **Stereo-seq**, **ST**, etc.
-  Scalable to large datasets with a high number of cells and spatial locations.
-  Performance of integrative analysis across modalities (scRNA-seq and SRT) and cell-type deconvolution for the non-single-cell resolution SRT data.

## Other Capabilities
-  Capability to produce not only the batch-corrected embeddings but also batch-corrected gene expression profiles for scRNA-seq data.
-  Capability to perform spatial mapping of reference scRNA-seq data points to the spatial locations of SRT data based on the learned co-embeddings and kNN.

## Installation
```
pip install step-kit
```

require python version 3.10+. Documentation and tutorials are available at [https://sggb0nd.github.io/step/](https://sggb0nd.github.io/step/)


## Contribution

We welcome contributions! Please see [`CONTRIBUTING.md`](./CONTRIBUTING.md) for more details!

## License

step is licensed under [LICENSE](./LICENSE)
