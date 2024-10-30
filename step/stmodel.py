import inspect
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple

import scanpy as sc
import seaborn as sns
from anndata import AnnData
from sklearn.cluster import KMeans

from step.functionality.st_funcmodel import stSmoother
from step.manager import logger
from step.models.clust import ClusterGeneric
from step.utils.dataset import StDataset
from step.utils.plotting import (plot_domain_summary,
                                 plot_domain_summary_single_ct,
                                 plot_single_domain_summary, spatial_plot)


class stModel:
    """
    stModel is the main class for spatial transcriptomics data.

    Attributes:
        adata: Annotated data object containing the gene expression data.
        dataset: StDataset object.
        _functional: stSmoother object.
    """

    def __init__(
        self,
        adata: Optional[AnnData] = None,
        file_path: Optional[str] = None,
        n_top_genes: Optional[int] = 2000,
        geneset_to_use: Optional[Sequence[str]] = None,
        batch_key: Optional[str] = None,
        layer_key: Optional[str] = None,
        coord_keys: Tuple[str, str] = ("array_row", "array_col"),
        log_transformed=False,
        module_dim=30,
        decoder_input_dim=None,
        hidden_dim=64,
        n_modules=32,
        model_checkpoint=None,
        edge_clip: float | Literal['visium'] = 'visium',
        logarithm_first=False,
        variational=True,
        n_glayers=4,
        hvg_method="seurat_v3",
        filtered=False,
        dispersion='gene',
        device=None,
        logarithm_after_hvgs=False,
        **kwargs,
    ):
        """
        Initialize the stModel object.

        Args:
            adata (Optional[AnnData]): Annotated data matrix with shape (n_obs, n_vars). If provided, it will be used to create the StDataset.
            file_path (Optional[str]): Path to the adata file. If provided, it will be used to create the StDataset.
            n_top_genes (Optional[int]): Number of top genes to select. Default is 2000.
            geneset_to_use (Optional[Sequence[str]]): List of genes to use. Default is None.
            batch_key (Optional[str]): Key for batch information in adata.obs. Default is None.
            layer_key (Optional[str]): Key for layer information in adata.layers. Default is None.
            coord_keys (Tuple[str, str]): Keys for spatial coordinates in adata.obsm. Default is ('array_row', 'array_col').
            log_transformed (bool): Whether the gene expression values are log-transformed. Default is False.
            module_dim (int): Dimensionality of the modules. Default is 30.
            decoder_input_dim (Optional[int]): Dimensionality of the decoder input. If None, it will be set to module_dim.
            hidden_dim (int): Dimensionality of the hidden layers. Default is 64.
            n_modules (int): Number of modules. Default is 32.
            model_checkpoint (Any): Model checkpoint to load. Default is None.
            edge_clip (int): Clip the adj edges to edge_clip. Default is 2.
            logarithm_first (bool): Whether to apply logarithm transformation before hvgs. Default is False.
            variational (bool): Whether to use variational inference. Default is False.
            n_glayers (int): Number of graph convolutional layers. Default is 4.
            hvg_method (str): Method to select highly variable genes. Default is "seurat_v3".
            filtered (bool): Whether the gene expression data (cells) are filtered. Default is False.
            **kwargs (Any): Additional keyword arguments.

        Raises:
            AssertionError: If neither adata nor file_path is provided.
        """

        adata_provided = adata is not None
        adata_file_path_provided = file_path is not None
        assert (adata_provided) or (
            adata_file_path_provided
        ), "adata or path to adata is required"
        if adata_provided:
            self.dataset = StDataset(
                adata=adata,
                n_top_genes=n_top_genes,
                geneset=geneset_to_use,
                coord_keys=coord_keys,
                layer_key=layer_key,
                batch_key=batch_key,
                log_transformed=log_transformed,
                filtered=filtered,
                logarithm_first=logarithm_first,
                hvg_method=hvg_method,
                logarithm_after_hvgs=logarithm_after_hvgs,
            )
        elif adata_file_path_provided:
            self.dataset = StDataset.read(
                path=file_path,
                n_top_genes=n_top_genes,
                geneset=geneset_to_use,
                layer_key=layer_key,
                batch_key=batch_key,
                coord_keys=coord_keys,
                log_transformed=log_transformed,
                filtered=filtered,
                logarithm_first=logarithm_first,
                logarithm_after_hvgs=logarithm_after_hvgs,
            )
        else:
            raise
        logger.info(self.dataset)

        self._functional = stSmoother(
            input_dim=self.dataset.gene_expr.shape[1],
            hidden_dim=hidden_dim,
            module_dim=module_dim,
            decoder_input_dim=decoder_input_dim,
            n_modules=n_modules,
            edge_clip=edge_clip,
            num_batches=self.dataset.num_batches,
            variational=variational,
            dispersion=dispersion,
            n_glayers=n_glayers,
            device=device,
            **kwargs,
        )
        if model_checkpoint is not None:
            try:
                self._functional.load_checkpoint(model_checkpoint)
            except Exception as e:
                logger.error(repr(e))
                logger.error(
                    "Load checkpoint failed, fall back to create a new instance"
                )
                # logger.info("Load checkpoint failed, fall back to create a new instance")

    def __getattr__(self, name):
        if hasattr(self._functional, name):
            attr = getattr(self._functional, name)
            if callable(attr):
                sig = inspect.signature(attr)

                def wrapper(*args, **kwargs):
                    if "adata" in sig.parameters:
                        kwargs["adata"] = kwargs.get("adata", self.adata)
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
        ======dataset======
        {self.dataset}
        ===================
        """

    @property
    def adata(self):
        return self.dataset.adata

    def cluster(
        self,
        adata: AnnData | None = None,
        n_clusters=3,
        use_rep="X_smoothed",
        key_added="domain",
        method="kmeans",
        seed=None,
    ):
        """
        Cluster the embedding of spatial transcriptomics.

        Args:
            adata (Optional[AnnData]): Annotated data matrix with shape (n_obs, n_vars). If provided, it will be used to perform clustering, and the result will be added to adata.obs; otherwise, it will use the adata in the stModel object.
            n_clusters (int): Number of clusters. Default is 3.
            use_rep (Optional[str]): Key for the representation to use. If None, it will use the default representation.
            key_added (str): Key to add to adata.obs. Default is "domain".
            method (str): Clustering method. Default is "kmeans".
            seed (Optional[int]): Random seed. Default is None.

        """
        if adata is None:
            adata = self.adata
        cluster = ClusterGeneric(method, seed)
        if use_rep is None:
            labels = cluster(n_clusters, adata.obsm[self.key_added])
        else:
            labels = cluster(n_clusters, adata.obsm[use_rep])
        adata.obs[key_added] = labels + 1
        adata.obs[key_added] = adata.obs[key_added].astype(str).astype(
            "category")  # type:ignore

    def sub_cluster(
        self,
        adata: Optional[AnnData],
        n_clusters=3,
        use_rep="X_smoothed",
        pre_key="domain",
        key_added="sub_domain",
    ):
        """
        Sub-cluster the clusters of spatial transcriptomics.

        Args:
            adata (Optional[AnnData]): Annotated data matrix with shape (n_obs, n_vars). If provided, it will be used to perform sub-clustering, and the result will be added to adata.obs; otherwise, it will use the adata in the stModel object.
            n_clusters (int): Number of clusters. Default is 3.
            use_rep (Optional[str]): Key for the representation to use. If None, it will use the default representation.
            pre_key (str): Key for the pre-clustered clusters in adata.obs. Default is "domain".
            key_added (str): Key to add to adata.obs. Default is "sub_domain".

        """
        if adata is None:
            adata = self.adata
        adata.obs[key_added] = adata.obs[pre_key].copy().astype(str)
        for clu in adata.obs[pre_key].unique():
            _adata = adata[adata.obs[pre_key] == clu, :]
            if use_rep is None:
                _labels = KMeans(n_clusters).fit_predict(
                    _adata.obsm[self.key_added])
            else:
                _labels = KMeans(n_clusters).fit_predict(_adata.obsm[use_rep])
            adata.obs.loc[adata.obs[pre_key] == clu, key_added] = [
                f"{clu}.{sub_clu + 1}" for sub_clu in _labels
            ]
        adata.obs[key_added] = adata.obs[key_added].astype(str).astype(
            "category")  # type:ignore

    def summarize_domain(
        self,
        cell_type_names,
        adata: Optional[AnnData] = None,
        domain_key="domain",
        average=True,
        obsm_key="deconv",
        figsize=(15, 5),
        show=True,
        save=False,
    ):
        """
        Summarize the domain of spatial transcriptomics.

        Args:
            cell_type_names (Sequence[str]): List of cell type names.
            adata (Optional[AnnData]): Annotated data matrix with shape (n_obs, n_vars). If provided, it will be used to perform sub-clustering, and the result will be added to adata.obs; otherwise, it will use the adata in the stModel object.
            domain_key (str): Key for the domain information in adata.obs. Default is "domain".
            average (bool): Whether to average the domain information. Default is True.
            obsm_key (str): Key for the deconvolution information in adata.obsm. Default is "deconv".
            figsize (Tuple[int, int]): Figure size. Default is (15, 5).
            show (bool): Whether to show the plot. Default is True.
            save (bool): Whether to save the plot. Default is False.

        """
        sns.set_style("white")
        if adata is None:
            adata = self.adata
        if average:
            fig = plot_domain_summary(
                adata, domain_key, cell_type_names, figsize=figsize, show=show
            )
        else:
            fig = plot_domain_summary_single_ct(
                adata, domain_key, cell_type_names, figsize=figsize, show=show
            )
        if save:
            fig.savefig(f"{obsm_key}_domain_summary.pdf", bbox_inches="tight")

    def summarize_single_domain(
        self,
        cell_type_names,
        adata: Optional[AnnData] = None,
        domain_key="domain",
        obsm_key="deconv",
        figsize=(15, 5),
        show=True,
        save=True,
    ):
        raise NotImplementedError
        sns.set_style("white")
        if adata is None:
            adata = self.adata
        adata.obs[cell_type_names] = adata.obsm[obsm_key]
        fig = plot_single_domain_summary(
            adata, domain_key, cell_type_names, figsize=figsize, show=show
        )
        if save:
            fig.savefig(f"{obsm_key}_single_domain_summary.pdf",
                        bbox_inches="tight")

    def save(self, path: str | Path = "."):
        from step.manager.save import Saver
        saver = Saver.get_instance(self.__class__.__name__)
        """Save the model and the data.

        Args:
            path (str): The path to save the model and the dataset.
        """
        saver.path = path
        saver.save(self._functional, self.adata, self.dataset)

    @classmethod
    def load(
        cls,
        path: str,
        adata: Optional[AnnData] = None,
        filepath: Optional[str] = None,
        config_name: str = "config.json",
        model_name: str = "model.pth",
    ):
        """Load the model and the data.

        Args:
            path (str): The path to load the model and the dataset.
            adata (Optional[AnnData]): Annotated data object containing the gene expression data.
            filepath (Optional[str]): Path to a file containing the gene expression data.
            config_name (str): The name of the config file.
            model_name (str): The name of the model file.

        Returns:
            scModel: The scModel object.
        """
        import json
        import os

        config = json.load(open(os.path.join(path, config_name), "r"))
        class_name = config["class_name"]
        adata_saved = config["save_adata"]
        saved_adata_path = config.get("adata_path", None)
        saved_adata = False
        if saved_adata_path and os.path.exists(saved_adata_path):
            saved_adata = True

        assert class_name == cls.__name__, f"Expected class name {cls.__name__}, got {class_name}"
        config["model_config"].pop("input_dim")
        config["model_config"].pop("num_batches", None)
        config["model_config"].pop("num_classes", None)
        config["dataset_config"]["geneset_to_use"] = config["dataset_config"].pop("geneset")

        if adata is not None or filepath is not None:
            if adata_saved and saved_adata:
                logger.warning("Using passed adata, but detected saved adata.")
            config["dataset_config"]["layer_key"] = config["dataset_config"].pop("received_layer_key", None)
            return cls(
                adata=adata,
                file_path=filepath,
                **config["model_config"],
                **config["dataset_config"],
                model_checkpoint=os.path.join(path, model_name),
            )
        else:
            assert adata_saved and saved_adata, "No adata provided and no saved adata detected"
            ds_config = config["dataset_config"]
            ds_config["log_transformed"] = True
            ds_config["logarithm_first"] = False
            ds_config["n_top_genes"] = None
            ds_config["filtered"] = True
            ds_config.pop("received_layer_key", None)
            return cls(
                filepath=saved_adata_path,
                **config["model_config"],
                **ds_config,
                model_checkpoint=os.path.join(path, model_name),
            )

    @property
    def model(self):
        return self._functional.model

    def spatial_plot(
        self,
        slide: str | int | None = None,
        with_images: bool = True,
        **kwargs,
    ):
        """Wrapper for plotting spatial feature plot with self-contained data.

        Args:
            slide (str | int | None): Slide name or index. Default is 0.
            with_images (bool): Whether to plot based on images which uses scanpy.pl.spatial. Default is True.
            **kwargs (Any): Additional keyword arguments for scanpy.pl.spatial or scanpy,pl.embedding

        Returns:
            matplotlib.figure.Figure: Figure.
        """
        adata = self.adata
        is_multi_batch = self.dataset.num_batches > 1
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
        )  # type:ignore
