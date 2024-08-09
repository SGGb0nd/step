import inspect
import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import seaborn as sns
from anndata import AnnData

from step.functionality.cross_funcmodel import CrossModalityNrmls
from step.manager import logger
from step.models.clust import ClusterGeneric
from step.utils.dataset import CrossDataset
from step.utils.plotting import plot_domain_summary, plot_single_domain_summary


class crossModel:
    """CrossModalityNrmls is the main class for integrating single-cell RNA-seq and spatial transcriptomics data.

    Attributes:
        adata: Single-cell RNA-seq data.
        st_adata: Spatial transcriptomics data.
        dataset: CrossDataset object.
        _functional: CrossModalityNrmls object.
    """

    def __init__(
        self,
        sc_adata: str | AnnData | Path,
        st_adata: str | AnnData | Path,
        class_key: str,
        n_top_genes: Optional[int] = 2000,
        geneset_to_use: Optional[Sequence[str]] = None,
        st_sample_name: Optional[str] = None,
        batch_key: Optional[str] = None,
        layer_key: Optional[str] = None,
        coord_keys: Tuple[str, str] = ("array_row", "array_col"),
        decoder_type: str = "zinb",
        log_transformed=False,
        module_dim=30,
        decoder_input_dim=None,
        hidden_dim=64,
        n_modules=32,
        model_checkpoint: str | Dict[str, str] | None = None,
        edge_clip=2,
        st_batch_key=None,
        variational=True,
        logarithm_first=False,
        hvg_method="seurat_v3",
        filtered=False,
        dispersion="batch-gene",
        device=None,
        **kwargs,
    ):
        """Initializes the Integration object.

        Args:
            sc_adata (str | AnnData): Single-cell RNA-seq data or path to the file containing the data.
            st_adata (str | AnnData): Spatial transcriptomics data or path to the file containing the data.
            class_key (str): Key in the AnnData object that specifies the cell type labels.
            n_top_genes (Optional[int]): Number of top genes to select. Default is 2000.
            geneset_to_use (Optional[Sequence[str]]): List of genes to use for integration. Default is None.
            st_sample_name (Optional[str]): Name of the st sample to use. Default is None.
            batch_key (Optional[str]): Key in the AnnData object that specifies the batch labels. Default is None.
            layer_key (Optional[str]): Key in the AnnData object that specifies the layer to use. Default is None.
            coord_keys (Tuple[str, str]): Tuple of keys in the AnnData object that specify the spatial coordinates. Default is ('array_row', 'array_col').
            log_transformed (bool): Whether the gene expression data is log-transformed. Default is False.
            module_dim (int): Dimension of the latent space. Default is 30.
            decoder_input_dim (Optional[int]): Dimension of the decoder input. Default is None.
            hidden_dim (int): Dimension of the hidden layers. Default is 64.
            n_modules (int): Number of modules in the model. Default is 32.
            model_checkpoint (Optional[str]): Path to the model checkpoint file. Default is None.
            edge_clip (int): Value to clip the edges of the latent space. Default is 2.
            st_batch_key (Optional[str]): Key in the AnnData object that specifies the batch labels for the st data. Default is None.
            variational (bool): Whether to use variational autoencoder. Default is True.
            logarithm_first (bool): Whether to apply logarithm to the gene expression data before normalization. Default is False.
            hvg_method (str): Method to use for selecting highly variable genes. Default is "seurat_v3".
            filtered (bool): Whether the gene expression data (cells) are filtered. Default is False.
            **kwargs: Additional keyword arguments to be passed to the CrossModalityNrmls constructor.
        """
        self.dataset = CrossDataset(
            sc_adata,
            st_adata,
            st_sample_names=st_sample_name,
            n_top_genes=n_top_genes,
            geneset=geneset_to_use,
            class_key=class_key,
            coord_keys=coord_keys,
            layer_key=layer_key,
            batch_key=batch_key,
            log_transformed=log_transformed,
            st_batch_key=st_batch_key,
            logarithm_first=logarithm_first,
            filtered=filtered,
            hvg_method=hvg_method,
        )
        logger.info(self.dataset)
        self._functional = CrossModalityNrmls(
            num_batches=self.dataset.num_batches,
            num_classes=self.dataset.num_classes,
            edge_clip=edge_clip,
            variational=variational,
            input_dim=self.dataset.gene_expr.shape[1],
            hidden_dim=hidden_dim,
            module_dim=module_dim,
            decoder_input_dim=decoder_input_dim,
            n_modules=n_modules,
            decoder_type=decoder_type,
            dispersion=dispersion,
            device=device,
            **kwargs,
        )
        if model_checkpoint is not None:
            try:
                self._functional.load_checkpoint(model_checkpoint)
            except Exception as e:
                logger.error(repr(e))
                import traceback
                logger.error(traceback.format_exc())
                logger.error(
                    "Load checkpoint failed, fall back to create a new instance"
                )

    def __getattr__(self, name):
        if hasattr(self._functional, name):
            attr = getattr(self._functional, name)
            if callable(attr):
                sig = inspect.signature(attr)

                def wrapper(*args, **kwargs):
                    if "adata" in sig.parameters:
                        kwargs["adata"] = kwargs.get("adata", self.adata)
                    if "st_adata" in sig.parameters:
                        kwargs["st_adata"] = kwargs.get(
                            "st_adata", self.st_adata)
                    if "dataset" in sig.parameters:
                        kwargs["dataset"] = kwargs.get("dataset", self.dataset)
                    return attr(*args, **kwargs)

                return wrapper
            else:
                return attr
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __repr__(self):
        # Display the adata objects and dataset and model config as members of the Integration object
        # notice that repr of adata is multiline, so we need to wrap it with =====(member)=====
        # to make it more readable
        return f"""\
        ====== adata ======
        {self.adata}
        ===================
        =====st_adata======
        {self.st_adata}
        ===================
        {self.dataset}
        """

    @property
    def adata(self) -> AnnData:
        """Returns AnnData object of the single-cell RNA-seq data."""
        return self.dataset.adata

    @property
    def st_adata(self) -> AnnData:
        """Returns AnnData object of the spatial transcriptomics data."""
        return self.dataset.st_adata

    def cluster(
        self,
        st_adata: Optional[AnnData] = None,
        n_clusters=3,
        use_rep='X_smoothed',
        key_added="domain",
        method="kmeans",
    ):
        """Cluster the embedding of spatial transcriptomics data.

        Args:
            st_adata (Optional[AnnData]): Spatial transcriptomics data. Default is None, which uses the spatial transcriptomics data provided during initialization.
            n_clusters (int): Number of clusters. Default is 3.
            use_rep (Optional[str]): Key of the embedding to use. Default is None.
            key_added (str): Key to add to the obs attribute of the spatial transcriptomics data. Default is "domain".
            method (str): Clustering method, either "kmeans" or "mclust". Default is "kmeans".
        """
        cluster = ClusterGeneric(method=method)
        if st_adata is None:
            st_adata = self.st_adata
        if use_rep is None:
            labels = cluster(n_clusters=n_clusters,
                             data=st_adata.obsm[self.key_added])
        else:
            labels = cluster(n_clusters=n_clusters,
                             data=st_adata.obsm[use_rep])
        st_adata.obs[key_added] = labels + 1
        st_adata.obs[key_added] = st_adata.obs[key_added].astype(
            "category"
        )  # type:ignore

    def sub_cluster(
        self,
        st_adata: Optional[AnnData] = None,
        n_clusters=3,
        use_rep='X_smoothed',
        pre_key="cluster",
        key_added="sub_domain",
        method="kmeans",
    ):
        """Sub-cluster the clusters of spatial transcriptomics data.

        Args:
            st_adata (Optional[AnnData]): Spatial transcriptomics data. Default is None, which uses the spatial transcriptomics data provided during initialization.
            n_clusters (int): Number of clusters. Default is 3.
            use_rep (Optional[str]): Key of the embedding to use. Default is None.
            pre_key (str): Key in the obs attribute of the spatial transcriptomics data that specifies the clusters. Default is "cluster".
            key_added (str): Key to add to the obs attribute of the spatial transcriptomics data. Default is "sub_domain".
            method (str): Clustering method, either "kmeans" or "mclust". Default is "kmeans".
        """
        if st_adata is None:
            st_adata = self.st_adata
        st_adata.obs[key_added] = st_adata.obs[pre_key].copy().astype(int)
        cluster = ClusterGeneric(method=method)
        for clu in st_adata.obs[pre_key].unique():
            _st_adata = st_adata[st_adata.obs[pre_key] == clu, :]
            if use_rep is None:
                _labels = cluster(
                    n_clusters, _st_adata.obsm[self.key_added]) + 1
            else:
                _labels = cluster(n_clusters, _st_adata.obsm[use_rep]) + 1
            st_adata.obs.loc[st_adata.obs[pre_key] == clu, key_added] = [
                f"{clu}.{sub_clu}" for sub_clu in _labels
            ]
        st_adata.obs[key_added] = st_adata.obs[key_added].astype(
            "category"
        )  # type:ignore

    def summarize_domain(
        self,
        cell_type_names,
        adata: Optional[AnnData] = None,
        domain_key="domain",
        obsm_key="deconv",
        figsize=(15, 5),
        show=True,
        save=True,
    ):
        """Summarize the average quantification of cell types in each domain.

        Args:
            cell_type_names (Sequence[str]): Names of the cell types.
            adata (Optional[AnnData]): Single-cell RNA-seq data. Default is None, which uses the single-cell RNA-seq data provided during initialization.
            domain_key (str): Key in the obs attribute of the spatial transcriptomics data that specifies the domain. Default is "domain".
            obsm_key (str): Key in the obsm attribute of the spatial transcriptomics data that specifies the cell type labels. Default is "deconv".
            figsize (Tuple[int, int]): Figure size. Default is (15, 5).
            show (bool): Whether to show the plot. Default is True.
            save (bool): Whether to save the plot. Default is True.
        """
        sns.set_style("white")
        if adata is None:
            adata = self.st_adata
        if not all(ct in adata.obs.columns for ct in cell_type_names):
            adata.obs[cell_type_names] = adata.obsm[obsm_key]
        fig = plot_domain_summary(
            adata, domain_key, cell_type_names, figsize=figsize, show=show
        )
        if save:
            fig.savefig(f"{obsm_key}_domain_summary.pdf", bbox_inches="tight")

    def summarize_single_domain(
        self,
        cell_type_names,
        domain: int | str,
        adata: Optional[AnnData] = None,
        domain_key="domain",
        obsm_key="deconv",
        figsize=(15, 5),
        show=True,
        save=True,
    ):
        """Summarize the quantification of cell types in each domain.

        Args:
            cell_type_names (Sequence[str]): Names of the cell types.
            adata (Optional[AnnData]): Single-cell RNA-seq data. Default is None, which uses the single-cell RNA-seq data provided during initialization.
            domain_key (str): Key in the obs attribute of the spatial transcriptomics data that specifies the domain. Default is "domain".
            obsm_key (str): Key in the obsm attribute of the spatial transcriptomics data that specifies the cell type labels. Default is "deconv".
            figsize (Tuple[int, int]): Figure size. Default is (15, 5).
            show (bool): Whether to show the plot. Default is True.
            save (bool): Whether to save the plot. Default is True.
        """
        sns.set_style("white")
        if adata is None:
            adata = self.st_adata
        if not all(ct in adata.obs.columns for ct in cell_type_names):
            adata.obs[cell_type_names] = adata.obsm[obsm_key]
        fig = plot_single_domain_summary(
            adata=adata,
            domain=domain,
            domain_key=domain_key,
            cell_type_names=cell_type_names,
            figsize=figsize,
            show=show,
        )
        if save:
            fig.savefig(f"{obsm_key}_single_domain_summary.pdf", bbox_inches="tight")

    def save(self, path: str = '.', save_adata: bool = False):
        """Save the model and the data.

        Args:
            path (str): The path to save the model and the dataset.
            save_adata (bool): Whether to save the adata object. Default is False.
        """
        from step.manager.save import Saver
        saver = Saver.get_instance(self.__class__.__name__)
        saver.path = path
        saver.save_adata = save_adata
        config = saver.save(self._functional, 
                            self.adata[self.adata.obs['modality'] == 'SC'], 
                            self.dataset)
        if saver.save_adata:
            self.dataset.st_adata.write_h5ad(f"{path}/st_adata.h5ad")
            config["st_adata_path"] = f"{path}/st_adata.h5ad"
            with json.open(f"{path}/config.json", "w") as f:
                json.dump(config, f)

    @classmethod
    def load(
        cls,
        path: str | Path,
        sc_adata: str | AnnData | None | Path = None,
        st_adata: str | AnnData | None | Path = None,
        config_name: str = "config.json",
        model_name: str = "model.pth",
        st_decoder_name: str = "st_decoder.pth",
        mixer_name: str = "mixer.pth",
    ):
        """Load the model and the data.

        Args:
            path (str): The path to load the model and the dataset.
            adata (Optional[AnnData]): Annotated data object containing the gene expression data.
            filepath (Optional[str]): Path to a file containing the gene expression data.
            config_name (str): The name of the config file.
            model_name (str): The name of the backbone model file.
            st_decoder_name (str): The name of the st_decoder file.
            mixer_name (str): The name of the mixer file.

        Returns:
            scModel: The scModel object.
        """
        path = Path(path)
        config = json.load(open(path / config_name, "r"))
        class_name = config["class_name"]
        adata_saved = config["save_adata"]
        saved_sc_adata_path = config.get("adata_path", None)
        saved_sc_adata = False
        if saved_sc_adata_path and os.path.exists(saved_sc_adata_path):
            saved_sc_adata = True
        saved_st_adata_path = config.get("st_adata_path", None)
        saved_st_adata = False
        if saved_st_adata_path and os.path.exists(saved_st_adata_path):
            saved_st_adata = True

        saved_adata = saved_sc_adata and saved_st_adata

        assert class_name == cls.__name__, f"Expected class name {cls.__name__}, got {class_name}"
        config["model_config"].pop("input_dim")
        config["model_config"].pop("num_batches", None)
        config["model_config"].pop("num_classes", None)
        config["dataset_config"]["geneset_to_use"] = config["dataset_config"].pop("geneset", None)
        st_decoder_config = config["model_config"].pop("st_decoder", None)
        mixer_config = config["model_config"].pop("mixer", None)
        model_checkpoint = dict(
            model=path / model_name,
            st_decoder=path / st_decoder_name,
            mixer=path / mixer_name,
            st_decoder_config=st_decoder_config,
            mixer_config=mixer_config,
        )

        if sc_adata is not None and st_adata is not None:
            if adata_saved and saved_adata:
                logger.warning("Using passed adata, but detected saved adata.")
            config["dataset_config"]["layer_key"] = config["dataset_config"].pop("received_layer_key", None)
            obj = cls(
                sc_adata=sc_adata,
                st_adata=st_adata,
                **config["model_config"],
                **config["dataset_config"],
                model_checkpoint=model_checkpoint,
            )
        else:
            assert adata_saved and saved_adata, "No adata provided and no saved adata detected"
            ds_config = config["dataset_config"]
            ds_config["log_transformed"] = True
            ds_config["logarithm_first"] = False
            ds_config["n_top_genes"] = None
            ds_config["filtered"] = True
            ds_config.pop("received_layer_key", None)
            obj = cls(
                sc_adata=saved_sc_adata_path,
                st_adata=saved_st_adata_path,
                **config["model_config"],
                **ds_config,
                model_checkpoint=model_checkpoint,
            )
        return obj

    def spatial_plot(
        self,
        slide: str | int | None = None,
        with_images: bool = True,
        **kwargs,
    ):
        import scanpy as sc

        from step.utils.plotting import spatial_plot
        """Wrapper for plotting spatial feature plot with self-contained data.

        Args:
            slide (str | int | None): Slide name or index. Default is 0.
            with_images (bool): Whether to plot based on images which uses scanpy.pl.spatial. Default is True.
            **kwargs (Any): Additional keyword arguments for scanpy.pl.spatial or scanpy,pl.embedding

        Returns:
            matplotlib.figure.Figure: Figure.
        """
        adata = self.st_adata
        is_multi_batch = (len(self.dataset.st_sample_names) > 1)
        if not is_multi_batch:
            return sc.pl.spatial(adata, **kwargs)

        if isinstance(slide, int):
            is_valid_index = slide < self.dataset.num_batches - 1
            assert is_valid_index and is_multi_batch, f"Slide index {slide} out of range"
            slide = self.dataset.batch_names[slide]  # type:ignore

        batch_key = self.dataset.batch_key
        return spatial_plot(
            adata, batch_key=batch_key, slide=slide, with_images=with_images,
            **kwargs,
        )

    @property
    def model(self):
        return self._functional.model
