# STEP: Spatial Transcriptomics Embedding Procedure
[![Docs](https://github.com/SGGb0nd/step/actions/workflows/mkdocs.yaml/badge.svg)](https://github.com/SGGb0nd/step/actions/workflows/mkdocs.yaml)
[![Pages](https://github.com/SGGb0nd/step/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/SGGb0nd/step/actions/workflows/pages/pages-build-deployment)
[![PyPI version](https://badge.fury.io/py/step-kit.svg)](https://badge.fury.io/py/step-kit)
[![DOI](http://img.shields.io/badge/DOI-10.1101/2024.04.15.589470-B31B1B.svg)](https://doi.org/10.1101/2024.04.15.589470)

![image](http://43.134.105.224/images/STEP_fig_1a.webp)
![image](http://43.134.105.224/images/STEP_fig_1b.webp)

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
-  Comprehensive `adata` processing by specifically desinged `BaseDataset` class and its view version `MaskedDataset` class for the SRT data, which can be easily integrated with the PyTorch `DataLoader` class for training and validation.
-  Low computational cost and high efficiency in processing large-scale SRT data: 4-8 GB GPU memory is sufficient for processing a dataset with 100,000 spatial locations with 2000+ sample-size in fast mode, and consumes less than 6 minutes for training in 2000 iterations (tested on NVIDIA RTX 3090).

## System Requiremtns
-  **Software Requirements**: Python 3.10+, CUDA 11.6+, PyTorch 1.13.1+, and dgl 1.1.3+.
-  **Hardware Requirements**: NVIDIA GPU with CUDA support (recommended), 8GB+ GPU memory. As possible as high RAM and CPU cores for storing and processing large-scale data (this is not required by STEP, but by the data itself).
## Installation

### Quick Installation (Recommended for Python 3.10, Linux, CUDA 11.7)

```bash
pip install step-kit[cu117]
```

This command installs STEP along with compatible versions of PyTorch and DGL.

### Basic Installation

```bash
pip install step-kit
```

⚠️ **Critical Note**: 
- This basic installation **does not** include PyTorch and DGL.
- STEP **will not function** without PyTorch and DGL properly installed.
- You must install these dependencies separately before using STEP.

### Manual Dependency Installation

If you need to install specific versions of PyTorch and DGL:

1. Install DGL (example for latest DGL compatible with PyTorch 2.3.x and CUDA 11.8):
   ```bash
   pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu118/repo.html
   ```

2. Install PyTorch (example for PyTorch 2.3.1 with CUDA 11.8):
   ```bash
   pip install torch==2.3.1+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
   ```

For more information on installing PyTorch and DGL, visit:
- PyTorch: https://pytorch.org/
- DGL: https://www.dgl.ai/

Remember to choose versions compatible with your system configuration and CUDA version (if applicable).

## Contribution

We welcome contributions! Please see [`CONTRIBUTING.md`](./CONTRIBUTING.md) for more details!

## License

step is licensed under [LICENSE](./LICENSE)

## Contact

If you have any questions, please feel free to contact us at [here](mailto:lilounan1997@gmail.com), or feel free to open an issue on this repository.

## Citation
The preprint of STEP is available at [bioRxiv](https://www.biorxiv.org/content/early/2024/04/20/2024.04.15.589470.full.pdf). If you use STEP in your research, please cite:

```bibtex
@article{Li2024.04.15.589470,
  title = {{{STEP}}: {{Spatial}} Transcriptomics Embedding Procedure for Multi-Scale Biological Heterogeneities Revelation in Multiple Samples},
  author = {Li, Lounan and Li, Zhong and Li, Yuanyuan and Yin, Xiao-ming and Xu, Xiaojiang},
  year = {2024},
  journal = {bioRxiv : the preprint server for biology},
  eprint = {https://www.biorxiv.org/content/early/2024/04/20/2024.04.15.589470.full.pdf},
  publisher = {Cold Spring Harbor Laboratory},
  doi = {10.1101/2024.04.15.589470},
}
```
