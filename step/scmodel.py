import inspect
from pathlib import Path
from typing import Optional, Sequence, Union

from anndata import AnnData

from step.functionality.sc_funcmodel import scMultiBatchNrmls, scSingleBatch
from step.manager import logger
from step.utils.dataset import ScDataset


class scModel:
    """
    scModel is the main class for single-cell RNA-seq data analysis.

    Attributes:
        adata: Annotated data object containing the gene expression data.
        dataset: ScDataset object.
        _functional: scSingleBatch or scMultiBatchNrmls object.
    """

    def __init__(
        self,
        adata: Optional[AnnData] = None,
        file_path: Optional[str] = None,
        n_top_genes: Optional[int] = 2000,
        geneset_to_use: Optional[Sequence[str]] = None,
        layer_key: Optional[str] = None,
        class_key: Optional[str] = None,
        batch_key: Optional[str] = None,
        log_transformed=False,
        module_dim=30,
        decoder_input_dim=None,
        hidden_dim=64,
        n_modules=32,
        model_checkpoint=None,
        beta=1e-2,
        decoder_type="zinb",
        dispersion="batch-gene",
        logarithm_first=False,
        hvg_method="seurat_v3",
        filtered=False,
        device=None,
        **kwargs,
    ):
        """
        Initialize the ScModel object.

        Args:
            adata (Optional[AnnData]): Annotated data object containing the gene expression data.
            file_path (Optional[str]): Path to a file containing the gene expression data.
            n_top_genes (Optional[int]): Number of top hvgs to use.
            geneset_to_use (Optional[Sequence[str]]): List of genes to consider.
            layer_key (Optional[str]): Key for accessing the layer in the adata object.
            class_key (Optional[str]): Key for accessing the class labels in the adata object.
            batch_key (Optional[str]): Key for accessing the batch information in the adata object.
            log_transformed (bool): Whether the gene expression data is log-transformed.
            module_dim (int): Dimension of the module.
            decoder_input_dim (Optional[int]): Dimension of the decoder input.
            hidden_dim (int): Dimension of the hidden layer.
            n_modules (int): Number of modules.
            model_checkpoint: Checkpoint for loading a pre-trained model.
            beta (float): Beta value for the loss function.
            decoder_type (str): Type of the decoder, either 'zinb' or 'nb'.
            logarithm_first (bool): Whether to apply logarithm transformation before other operations.
            hvg_method (str): Method for selecting highly variable genes.
            filtered (bool): Whether the gene expression data (cells) are filtered.
            **kwargs: Additional keyword arguments.

        Raises:
            AssertionError: If neither adata nor file_path is provided.
        """

        adata_provided = adata is not None
        adata_file_path_provided = file_path is not None
        assert (adata_provided) or (
            adata_file_path_provided
        ), "adata or path to adata is required"
        if adata_provided:
            self.dataset = ScDataset(
                adata=adata,
                n_top_genes=n_top_genes,
                geneset=geneset_to_use,
                layer_key=layer_key,
                class_key=class_key,
                batch_key=batch_key,
                log_transformed=log_transformed,
                logarithm_first=logarithm_first,
                filtered=filtered,
                hvg_method=hvg_method,
            )
        elif adata_file_path_provided:
            self.dataset = ScDataset.read(
                path=file_path,
                n_top_genes=n_top_genes,
                geneset=geneset_to_use,
                layer_key=layer_key,
                class_key=class_key,
                batch_key=batch_key,
                log_transformed=log_transformed,
                logarithm_first=logarithm_first,
                filtered=filtered,
            )
        else:
            raise
        logger.info(self.dataset)
        self._model_config = {}  # TODO: add config
        self._functional = self._select_functional(
            module_dim,
            hidden_dim,
            n_modules,
            decoder_input_dim,
            model_checkpoint,
            beta,
            decoder_type=decoder_type,
            dispersion=dispersion,
            device=device,
            **kwargs,
        )

    def _select_functional(
        self,
        module_dim,
        hidden_dim,
        n_modules,
        decoder_input_dim,
        model_checkpoint,
        beta,
        decoder_type,
        dispersion,
        device=None,
        **kwargs,
    ):
        num_batches = self.dataset.num_batches
        if num_batches == 1:
            functional = scSingleBatch(
                input_dim=self.dataset.gene_expr.shape[1],
                hidden_dim=hidden_dim,
                module_dim=module_dim,
                decoder_input_dim=decoder_input_dim,
                n_modules=n_modules,
                decoder_type=decoder_type,
                device=device,
                n_glayers=None,
                **kwargs,
            )
        else:
            functional = scMultiBatchNrmls(
                num_batches=num_batches,
                input_dim=self.dataset.gene_expr.shape[1],
                hidden_dim=hidden_dim,
                module_dim=module_dim,
                decoder_input_dim=decoder_input_dim,
                n_modules=n_modules,
                decoder_type=decoder_type,
                dispersion=dispersion,
                n_glayers=None,
                device=device,
                **kwargs,
            )
        self.load_checkpoint(model_checkpoint, functional)
        functional.set_beta(beta)
        return functional

    def load_checkpoint(
        self, model_checkpoint, functional: Union[scSingleBatch, scMultiBatchNrmls]
    ):
        if model_checkpoint is not None:
            try:
                functional.load_checkpoint(model_checkpoint)
            except Exception as e:
                logger.info(repr(e))
                logger.info("Load checkpoint failed, fall back to create a new instance")

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
        {self.__class__.__name__} object with members:
        ====== adata ======
        {self.adata}
        ===================
        {self.dataset}
        """

    @property
    def adata(self):
        return self.dataset.adata

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
