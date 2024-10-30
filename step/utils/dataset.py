import os
import warnings
from enum import Enum
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from torch.utils.data import Dataset

from step.manager import logger

warnings.filterwarnings("ignore")

PROCESS_FIELDS = (
    "adata",
    "is_human",
    "layer_key",
    "batch_key",
    "geneset",
    "log_transformed",
    "logarithm_first",
    "n_top_genes",
    "filtered",
    "hvg_method",
)

READ_FIELDS = (
    "adata",
    "log_transformed",
    "logarithm_first",
    "geneset",
    "n_top_genes",
    "layer_key",
    "batch_key",
)

RECIEVE_FIELDS = (
    "gene_expr",
    "adata",
    "batch_label",
    "_batch_codes",
    "layer_key",
)


class ScMode(Enum):
    """
    Enum class for single-cell dataset output modes.

    Attributes:
        single_batch (list): List of fields for single-batch mode.
        multi_batches (list): List of fields for multi-batch mode.
        multi_batches_with_ct (list): List of fields for multi-batch mode with class labels.
    """
    single_batch = ["gene_expr"]
    multi_batches = ["gene_expr", "batch_label"]
    multi_batches_with_ct = ["gene_expr", "class_label", "batch_label"]

    def __str__(self):
        return f"{self.name}: {self.value}"


class StMode(Enum):
    """Enum class for spatial transcriptomics dataset output modes.

    Attributes:
        single_batch (list): List of fields for single-batch mode.
        multi_batches (list): List of fields for multi-batch mode.
        with_rep (list): List of fields for mode with representation.
        with_domains (list): List of fields for mode with domains.
    """
    single_batch = ["gene_expr"]
    multi_batches = ["gene_expr", "batch_label"]
    with_rep = ["gene_expr", "rep", "batch_label"]
    with_domains = ["gene_expr", "rep", "domains", "batch_label"]

    def __str__(self):
        return f"{self.name}: {self.value}"


class MaskedDataset(Dataset):
    """
    A dataset class that represents a masked subset of an original dataset.

    Attributes:
        original_dataset (BaseDataset): The original dataset.
        _indices (ndarray): The indices of the samples in the subset.
        _mask (ndarray): The mask indicating which samples to include in the subset.

    """

    def __init__(self, original_dataset, mask):
        """
        Initialize the MaskedDataset object.

        Args:
            original_dataset (BaseDataset): The original dataset.
            mask (ndarray): The mask indicating which samples to include in the subset.

        """
        self.original_dataset: BaseDataset = original_dataset
        self._indices = np.nonzero(mask)[0]
        self._mask = mask

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        return self.original_dataset[self._indices[idx]]

    def __getattr__(self, name):
        attr = self._get_original_info(name)
        if attr is not None:
            return attr
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def _get_original_info(self, attr, default_val=None):
        attr_val = getattr(self.original_dataset, attr, None)
        if attr_val is None:
            return default_val
        if attr in self.original_dataset.mode.value:
            attr_val = attr_val[self._indices]
        return attr_val

    def set_mode(self, mode=None):
        """Sets the mode of the original dataset.

        Args:
            mode (str, optional): The mode to set. If not provided, the mode will be set to the default mode.

        Returns:
            str: The previous mode of the original dataset.
        """
        if mode is not None:
            return self.original_dataset.set_mode(mode)
        return self.original_dataset.set_mode()

    def subset(self, key: Iterable = True, col=None, exclude=True):
        """Returns a new MaskedDataset object representing a subset of the current subset.

        Args:
            key (str or Iterable, optional): The key or keys to filter the subset. If not provided, all samples are included.
            col (str, optional): The column name to use for filtering. If not provided, the batch key of the original dataset is used.
            exclude (bool, optional): If True, the subset will exclude the samples matching the key(s). If False, the subset will include only the samples matching the key(s).

        Returns:
            MaskedDataset: A new MaskedDataset object representing the subset.
        """
        if col is None:
            col = self._get_original_info("batch_key")
        if not isinstance(key, str) and isinstance(key, Iterable):
            ind = self.adata.obs[col].isin(key)
        else:
            ind = self.adata.obs[col] == key

        mask = ~ind if exclude else ind
        return MaskedDataset(self, mask.values)

    @property
    def mode(self):
        """
        The mode of the original dataset.

        Returns:
            str: The mode of the original dataset.
        """
        return self.original_dataset.mode

    @mode.setter
    def mode(self, value=None):
        self.set_mode(mode=value)

    @property
    def adata(self):
        return self.original_dataset.adata[self._indices]

    @property
    def masked_num_batches(self):
        return len(self.adata.obs[self.batch_key].value_counts())

    @property
    def masked_num_classes(self):
        return len(self.adata.obs[self.class_key].value_counts())

    def __repr__(self) -> str:
        repr_str = "==============Sub-Dataset Info==============\n"
        repr_str += (
            f"Batch key: {self._get_original_info('batch_key')} \n"
            + f"Class key: {self._get_original_info('class_key')} \n"
            + f"Number of Batches: {self.masked_num_batches}/{self._get_original_info('num_batches', 1)} \n"
            + f"Number of Classes: {self.masked_num_classes}/{self._get_original_info('num_classes', 'None')} \n"
        )
        for i, item in enumerate(self.mode.value):
            item_shape = tuple(self._get_original_info(item).shape)
            item_name = " ".join([w.capitalize() for w in item.split("_")])
            repr_str += f"{item_name}: {item_shape}"
            if i != len(self.mode.value) - 1:
                repr_str += "\n"
        repr_str += "\n============================================"
        return repr_str


class BaseDataset(Dataset):
    """
    Base dataset class for single-cell and spatial datasets.

    Attributes:
        read_fields (tuple): Tuple of fields to be read.
        process_fields (tuple): Tuple of fields to be processed.
        recieve_fields (tuple): Tuple of fields to be received.
        read_folder (function): Function to read the dataset from a folder.
        is_human (bool): Flag indicating if the dataset is from human.
        n_top_genes (int): Number of top genes to be selected.
        geneset (str): Geneset to be used.
        num_batches (int): Number of batches in the dataset.
        num_classes (int): Number of classes in the dataset.
        layer_key (str): Key for the layer in the data.
        batch_key (str): Key for the batch in the data.
        logarithm_first (bool): Flag indicating if logarithm transformation is applied first.
        hvg_method (str): The method for selecting highly variable genes.
        filtered (bool): Flag indicating if the dataset is filtered.
        log_transformed (bool): Flag indicating if the dataset is log-transformed.
        adata (AnnData): The annotated data matrix.
        mode (ScMode or StMode): Mode of the dataset for output fields.

    """

    read_fields = READ_FIELDS
    process_fields = PROCESS_FIELDS
    recieve_fields = RECIEVE_FIELDS
    def read_folder(_): return None  # noqa

    def __init__(
        self,
        adata: AnnData,
        filtered=False,
        log_transformed=False,
        logarithm_first=False,
        hvg_method="seurat_v3",
        geneset=None,
        n_top_genes: Optional[int] = 2000,
        layer_key: Optional[str] = None,
        batch_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the BaseDataset object.

        Args:
            adata (AnnData): The annotated data matrix.
            filtered (bool, optional): Flag indicating if the dataset is filtered. Defaults to False.
            log_transformed (bool, optional): Flag indicating if the dataset is log-transformed. Defaults to False.
            logarithm_first (bool, optional): Flag indicating if logarithm transformation is applied first. Defaults to False.
            geneset (Optional[str], optional): The geneset to be used. Defaults to None.
            n_top_genes (Optional[int], optional): The number of top genes to be selected. Defaults to 2000.
            layer_key (Optional[str], optional): The key for the layer in the data. Defaults to None.
            batch_key (Optional[str], optional): The key for the batch in the data. Defaults to None.
            **kwargs: Additional keyword arguments.

        """
        super(BaseDataset, self).__init__()
        assert adata is not None
        self.is_human = adata.var_names.str.contains("^MT", regex=True).any()
        self.n_top_genes = n_top_genes
        self.geneset = geneset
        if n_top_genes is None and geneset is None:
            logger.warning("Using all genes")
            self.geneset = adata.var_names
        self.num_batches = 1
        self.num_classes = None
        self.received_layer_key = layer_key
        self.layer_key = layer_key
        self.batch_key = batch_key
        self.logarithm_first = logarithm_first
        self.hvg_method = hvg_method
        self._batch_codes = None
        if self.batch_key is not None:
            assert self.batch_key in adata.obs.columns
            self.num_batches = len(
                adata.obs[self.batch_key].value_counts()
            )  # type:ignore
            adata.obs[self.batch_key] = adata.obs[self.batch_key].astype(
                "category")
        self.filtered = filtered
        self.log_transformed = log_transformed
        self.adata = adata

        process_config = {k: getattr(self, k) for k in self.process_fields}
        process_config = {**process_config, **kwargs}
        self.process(**process_config)

    @classmethod
    def read(
        cls,
        path: str,
        log_transformed=False,
        logarithm_first=False,
        filtered=False,
        geneset=None,
        batch_key: Optional[str] = None,
        n_top_genes: Optional[int] = 2000,
        layer_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Read the dataset from a file or folder.

        Args:
            path (str): The path to the file or folder.
            log_transformed (bool, optional): Whether the data is log-transformed. Defaults to False.
            logarithm_first (bool, optional): Whether to apply logarithm transformation first. Defaults to False.
            filtered (bool, optional): Whether the data is filtered. Defaults to False.
            geneset (Optional[str], optional): The geneset to be used. Defaults to None.
            batch_key (Optional[str], optional): The key for the batch in the data. Defaults to None.
            n_top_genes (Optional[int], optional): The number of top genes to be selected. Defaults to 2000.
            layer_key (Optional[str], optional): The key for the layer in the data. Defaults to None.
            **kwargs: Additional keyword arguments.

        Raises:
            Exception: If no valid path is provided.

        Returns:
            BaseDataset: The dataset object.

        """
        read_fields = cls.read_fields
        space_ranger_path = None
        h5ad_filepath = None
        if os.path.isdir(path):
            space_ranger_path = path
        elif os.path.splitext(path)[1] == ".h5ad":
            h5ad_filepath = path
        else:
            raise Exception(
                'No valid path was provided, neither cellranger path nor ".h5ad" file path'
            )
        if space_ranger_path:
            adata = cls.read_folder(space_ranger_path)
        elif h5ad_filepath is not None:
            adata = sc.read_h5ad(h5ad_filepath)
        else:
            raise FileNotFoundError

        local_items = locals()
        read_config = {k: local_items.get(
            k, kwargs.get(k, None)) for k in read_fields}
        return cls(**read_config)

    def process(self, **process_config):
        """
        Process the dataset.

        Args:
            **process_config: Keyword arguments for the processing configuration.

        """
        processed = _process_adata(**process_config)
        for key in self.recieve_fields:
            setattr(self, key, processed[key])

        self.log_transformed = True

    def get(self, batch=None, attr: Optional[str] = "gene_expr"):
        """Get the data for a specific batch.

        Args:
            batch: The batch identifier.
            attr (Optional[str], optional): The attribute to be retrieved. Defaults to 'gene_expr'.

        Returns:
            Union[np.ndarray, None]: The data for the specified batch.

        """
        index = np.arange(self.adata.n_obs, dtype=int)
        if batch is not None:
            batch_code = self._batch_codes[batch]
            mask = self.batch_label == batch_code
            index = index[mask]
        if attr is None:
            return index
        attr_val = getattr(self, attr, None)
        if attr_val is not None:
            return attr_val[index]
        return None

    def switch_layer(self, layer_key):
        assert layer_key in self.adata.layers.keys()
        data = self.adata.layers[layer_key]
        assert data.shape[0] == self.adata.n_obs, "Data shape 0 mismatch"
        assert data.shape[1] == self.n_genes, "Data shape 1 mismatch"
        self.gene_expr = torch.from_numpy(data)

    def subset(
        self,
        key=True,
        col=None,
        exclude=True,
    ):
        """
        Create a subset of the dataset based on a condition.

        Args:
            key: The condition for selecting the subset.
            col (Optional[str], optional): The column to be used for the condition. Defaults to None.
            exclude (bool, optional): Whether to exclude the selected subset. Defaults to True.

        Returns:
            MaskedDataset: The subset of the dataset.

        """
        if col is None:
            col = self.batch_key
        if not isinstance(key, str) and isinstance(key, Iterable):
            ind = self.adata.obs[col].isin(key)
        else:
            ind = self.adata.obs[col] == key

        mask = ~ind if exclude else ind
        return MaskedDataset(self, mask.values)

    @property
    def batch_codes(self) -> Optional[pd.DataFrame]:
        """
        Get the batch codes.

        Returns:
            Optional[pd.DataFrame]: The batch codes.

        """
        return self._batch_codes

    @property
    def batch_names(self) -> pd.Index | None:
        """
        Get the batch names.

        Returns:
            Optional[pd.DataFrame]: The batch names.

        """
        if self._batch_codes is not None:
            return self._batch_codes.index

    @property
    def class_codes(self) -> Optional[pd.DataFrame]:
        """
        Get the class codes.

        Returns:
            Optional[pd.DataFrame]: The class codes.

        """
        return self._class_codes

    @property
    def n_genes(self):
        """Get the number of genes.

        Returns:
            int: The number of genes.

        """
        return self.gene_expr.shape[1]

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, index):
        items = [getattr(self, item) for item in self.mode.value]
        if len(items) == 1:
            return items[0][index]
        return tuple([item[index] for item in items])

    def __repr__(self) -> str:
        repr_str = "================Dataset Info================\n"
        repr_str += (
            f"Batch key: {getattr(self, 'batch_key', None)} \n"
            + f"Class key: {getattr(self, 'class_key', None)} \n"
            + f"Number of Batches: {getattr(self, 'num_batches', 1)} \n"
            + f"Number of Classes: {getattr(self, 'num_classes', None)} \n"
        )
        for i, item in enumerate(self.mode.value):
            item_shape = tuple(getattr(self, item).shape)
            item_name = " ".join([w.capitalize() for w in item.split("_")])
            repr_str += f"{item_name}: {item_shape}"
            if i != len(self.mode.value) - 1:
                repr_str += "\n"
        repr_str += "\n============================================"
        return repr_str


class ScDataset(BaseDataset):
    """
    Single-cell dataset class.

    Attributes:
        class_key (str): Key for the class labels in the data.
        read_fields (tuple): Tuple of fields to be read.
        recieve_fields (tuple): Tuple of fields to be received.
        process_fields (tuple): Tuple of fields to be processed.
        read_folder (function): Function to read the dataset from a folder.
        num_classes (int): Number of classes in the data.
        mode (ScMode): Mode of the dataset.

    """

    read_fields = (*READ_FIELDS, "class_key")
    recieve_fields = (
        *RECIEVE_FIELDS,
        "class_label",
        "_class_codes",
    )
    process_fields = (*PROCESS_FIELDS, "class_key")
    read_folder = sc.read_10x_h5

    def __init__(self, class_key=None, **kwargs):
        """
        Initialize the ScDataset object.

        Args:
            class_key (Optional[str], optional): The key for the class labels in the data. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.class_key = class_key
        super(ScDataset, self).__init__(**kwargs)
        if self.class_key is not None:
            assert self.class_key in self.adata.obs.columns
            self.num_classes = len(
                self.adata.obs[self.class_key].value_counts()
            )  # type:ignore
        if self.num_batches > 1:
            self.mode = ScMode.multi_batches
        else:
            self.mode = ScMode.single_batch

    def set_mode(self, mode=ScMode.multi_batches_with_ct):
        """
        Set the mode of the dataset.

        Args:
            mode (Union[str, ScMode], optional): The mode to be set. Defaults to ScMode.multi_batches_with_ct.

        Returns:
            ScMode: The previous mode.

        """
        prev_mode = self.mode
        if isinstance(mode, str):
            mode = getattr(ScMode, mode, None)
            assert mode is not None, f"{mode} not found in ScMode"
        logger.info(f"Current Mode: {mode}")
        self.mode = mode
        return prev_mode


class StDataset(BaseDataset):
    """
    Spatial dataset class.

    Attributes:
        coord_keys (list): List of keys for the spatial coordinates in the data.
        read_fields (tuple): Tuple of fields to be read.
        recieve_fields (tuple): Tuple of fields to be received.
        process_fields (tuple): Tuple of fields to be processed.
        read_folder (function): Function to read the dataset from a folder.
        rep (object): Representation object.
        domains (object): Domains object.
        mode (StMode): Mode of the dataset.

    """

    read_folder = sc.read_visium
    read_fields = (*READ_FIELDS, "coord_keys")

    def __init__(
        self,
        coord_keys: Tuple[str, str] = ("array_row", "array_col"),
        rep=None,
        domains=None,
        **kwargs,
    ):
        """
        Initialize the StDataset object.

        Args:
            coord_keys (Tuple[str, str], optional): The keys for the spatial coordinates in the data. Defaults to ("array_row", "array_col").
            rep (Optional[object], optional): The representation object. Defaults to None.
            domains (Optional[object], optional): The domains object. Defaults to None.
            **kwargs: Additional keyword arguments.

        """
        self.coord_keys = list(coord_keys)

        super(StDataset, self).__init__(**kwargs)
        self.adata.obs[["array_row", "array_col"]
                       ] = self.adata.obs[self.coord_keys]
        if "spatial" not in self.adata.obsm_keys():
            self.adata.obsm["spatial"] = self.adata.obs[self.coord_keys].values
        self.rep = rep
        self.domains = domains
        self.mode = StMode.single_batch
        if self.num_batches > 1:
            self.mode = StMode.multi_batches
        if self.rep is not None:
            self.mode = StMode.with_rep
        if self.domains is not None:
            self.mode = StMode.with_domains
            self.mode = StMode.with_rep
        if self.domains is not None:
            self.mode = StMode.with_domains
        logger.info("Dataset Done")

    def set_domains(self, domains):
        assert len(domains) == self.__len__()
        self.domains = domains
        self.mode = StMode.with_domains

    def set_prior_coef(self, prior_coef):
        setattr(self, "prior_coef", prior_coef)

    def set_mode(self, mode=StMode.multi_batches):
        prev_mode = self.mode
        if isinstance(mode, str):
            mode = getattr(StMode, mode, None)
            assert mode is not None, f"{mode} not found in StMode"
        self.mode = mode
        return prev_mode


class CrossDataset(BaseDataset):
    """
    A dataset class for cross-modal single-cell and spatial transcriptomics data.

    Attributes:
        process_fields (tuple): Tuple of fields to be processed.
        recieve_fields (tuple): Tuple of fields to be received.
        class_key (str): Key for the class labels in the single-cell data.
        coord_keys (list): List of keys for the spatial coordinates in the spatial transcriptomics data.
        st_sample_names (list): List of sample names in the spatial transcriptomics data.
        batch_key (str): Key for the batch information in the single-cell data.
        num_classes (int): Number of classes in the single-cell data.
        st_adata (AnnData): Spatial transcriptomics AnnData object.
        rep (object): Representation object.
        _integrated (bool): Flag indicating if the dataset is integrated.
        mode (ScMode or StMode): Mode of the dataset.

    """

    process_fields = (
        *PROCESS_FIELDS,
        "class_key",
    )
    recieve_fields = (
        *RECIEVE_FIELDS,
        "class_label",
        "_class_codes",
    )

    def __init__(
        self,
        sc_adata: str | AnnData,
        st_adata: str | AnnData,
        class_key: str,
        coord_keys: Tuple[str, str] | str = ("array_row", "array_col"),
        batch_key=None,
        st_sample_names=None,
        st_batch_key=None,
        **kwargs,
    ):
        """
        Initialize the CrossDataset object.

        Args:
            sc_adata (Union[str, AnnData]): The single-cell AnnData object or the path to the file.
            st_adata (Union[str, AnnData]): The spatial transcriptomics AnnData object or the path to the file.
            class_key (str): The key for the class labels in the single-cell data.
            coord_keys (Tuple[str, str], optional): The keys for the spatial coordinates in the spatial transcriptomics data. Defaults to ("array_row", "array_col").
            batch_key (Optional[str], optional): The key for the batch information in the single-cell data. Defaults to None.
            st_sample_names (Optional[Iterable], optional): The sample names in the spatial transcriptomics data. Defaults to None.
            st_batch_key (Optional[str], optional): The key for the batch information in the spatial transcriptomics data. Defaults to None.
            **kwargs: Additional keyword arguments.

        """
        sc_adata = _read_sc(sc_adata)
        st_adata = _read_st(st_adata)
        sc_adata.var_names_make_unique()
        st_adata.var_names_make_unique()

        self.st_batch_key = st_batch_key
        self.class_key = class_key
        self.coord_keys = list(coord_keys)
        is_obsm = isinstance(coord_keys, str)
        try:
            if is_obsm:
                st_adata.obsm[self.coord_keys] = st_adata.obsm[self.coord_keys].astype(
                    float)
                st_adata.obs[["array_row", "array_col"]
                             ] = st_adata.obsm[self.coord_keys]
            else:
                st_adata.obs[["array_row", "array_col"]
                             ] = st_adata.obs[self.coord_keys]
        except KeyError:
            raise KeyError(
                f"Coordinate keys {coord_keys} not found in adata.obs")
        st_adata.obs[["array_row", "array_col"]] = st_adata.obs[["array_row", "array_col"]].astype(float)
        _assert_dtype(st_adata.obs[["array_row", "array_col"]].values)

        (
            adata,
            self.st_sample_names,
            self.batch_key,
        ) = _merge_sc_st_adata(
            sc_adata=sc_adata,
            st_adata=st_adata,
            st_sample_names=st_sample_names,
            batch_key=batch_key,
            st_batch_key=st_batch_key,
        )
        super(CrossDataset, self).__init__(
            adata=adata, batch_key=self.batch_key, **kwargs
        )

        if self.class_key is not None:
            assert self.class_key in sc_adata.obs.columns
            self.num_classes = len(
                sc_adata.obs[self.class_key].value_counts()
            )  # type:ignore

        filtermt = kwargs.get("filtermt", True)
        if filtermt:
            st_adata = _filter_mt(st_adata, is_human=self.is_human)

        st_adata = st_adata[self.st_obs_names]
        assert (
            st_adata.n_obs
            == self.adata[
                self.adata.obs[self.batch_key].isin(self.st_sample_list)
            ].n_obs
        ), "Unmatched number of samples of st data and merged data"

        st_adata.raw = st_adata
        st_adata.layers[self.layer_key] = st_adata.X.copy()
        st_adata = st_adata[:, self.adata.var_names]
        self.st_adata = st_adata.copy()
        del st_adata

        self.rep = None
        self._integrated = False
        self.mode = ScMode.multi_batches
        self.process_fields = (*self.process_fields, "st_batch_key")
        logger.info("Dataset Done")

    def set_rep(self, rep):
        """
        Set the representation for the dataset.

        Args:
            rep: The representation to be set.

        Returns:
            None
        """
        self.rep = rep
        self._integrated = True
        self.mode = StMode.with_rep

    def set_mode(self, mode=ScMode.multi_batches_with_ct):
        """
        Sets the mode of the dataset.

        Args:
            mode (ScMode or str): The mode to set. If a string is provided, it will be converted to the corresponding ScMode enum value.

        Returns:
            ScMode: The previous mode before setting the new mode.
        """
        prev_mode = self.mode
        if isinstance(mode, str):
            mode = getattr(ScMode, mode, None)
            assert mode is not None, f"{mode} not found in ScMode"
        self.mode = mode
        return prev_mode

    @classmethod
    def read(cls):
        raise NotImplementedError

    def subset_st(self, adata=None, domains=None) -> MaskedDataset:
        """Creates a subset specific to the spatial transcriptomics dataset.

        Args:
            rep (Optional[str], optional): The representation to be used. Defaults to None.
            adata (Optional[AnnData], optional): The annotated data matrix. Defaults to None.
            domains (Optional[np.ndarray], optional): The domains to be used. Defaults to None.

        Returns:
            MaskedDataset: The sole spatial transcriptomics dataset.

        """
        assert self.rep is not None
        if adata is None:
            adata = self.adata
        assert self.st_sample_names is not None
        mask = (adata.obs[self.batch_key].isin(self.st_sample_list)).values
        self.set_mode(StMode.with_rep)
        if domains is not None:
            setattr(self, "domains", torch.zeros(
                len(self), dtype=torch.int8) - 1)
            ind = np.nonzero(mask)[0]
            self.domains[ind] = domains
            self.set_mode(StMode.with_domains)
        return MaskedDataset(self, mask)

    @property
    def st_batch_codes(self) -> pd.Series:
        """
        Get the batch codes for the spatial transcriptomics dataset.

        Returns:
            np.ndarray: The batch codes for the spatial transcriptomics dataset.

        """
        return self._batch_codes[self.st_sample_list]

    @property
    def st_sample_list(self):
        """
        Get the sample list for the spatial transcriptomics dataset.

        Returns:
            list: The sample list for the spatial transcriptomics dataset.

        """
        return list(self.st_sample_names)

    @property
    def num_st_batches(self):
        """
        Get the number of batches for the spatial transcriptomics dataset.

        Returns:
            int: The number of batches for the spatial transcriptomics dataset.
        """
        return len(self.st_sample_names)

    @property
    def st_obs_names(self):
        obs_names = self.adata[
            self.adata.obs[self.batch_key].isin(
                self.st_sample_list
            )
        ].obs_names
        return obs_names.to_series().apply(lambda x: x[:-3]).to_list()


def _process_adata(
    adata: AnnData,
    is_human,
    layer_key=None,
    batch_key=None,
    class_key=None,
    geneset=None,
    filtered=False,
    filtermt=True,
    log_transformed=False,
    logarithm_first=False,
    hvg_method="seurat_v3",
    n_top_genes: Optional[int] = 3000,
    logarithm_after_hvgs=False,
) -> dict:
    """
    Process the AnnData object and extract relevant information for downstream analysis.

    Args:
        adata (AnnData): The input AnnData object.
        is_human (bool): Flag indicating whether the data is human or not.
        layer_key (str, optional): The key for the layer in the AnnData object. Defaults to None.
        batch_key (str, optional): The key for the batch information in the AnnData object. Defaults to None.
        class_key (str, optional): The key for the class information in the AnnData object. Defaults to None.
        geneset (list, optional): List of genes to subset the data. Defaults to None.
        filtered (bool, optional): Flag indicating whether the data is already filtered. Defaults to False.
        filtermt (bool, optional): Flag indicating whether to filter mitochondrial genes. Defaults to True.
        log_transformed (bool, optional): Flag indicating whether the data is already log-transformed. Defaults to False.
        logarithm_first (bool, optional): Flag indicating whether to perform log transformation before other preprocessing steps. Defaults to False.
        n_top_genes (int, optional): Number of top genes to select. Defaults to 3000.
        normalize (bool, optional): Flag indicating whether to normalize the data. Defaults to False.

    Returns:
        dict: A dictionary containing the processed data and relevant information.
            - gene_expr: The gene expression data.
            - adata: The processed AnnData object.
            - batch_label: The batch labels.
            - class_label: The class labels.
            - _batch_codes: The batch codes.
            - _class_codes: The class codes.
            - layer_key: The key for the layer in the AnnData object.
    """
    if not filtered:
        adata.var_names_make_unique()
        if filtermt:
            adata = _filter_mt(adata, is_human=is_human)
        sc.pp.filter_cells(
            adata,
            min_genes=10,
        )
    if not log_transformed and logarithm_first:
        sc.pp.log1p(adata, layer=layer_key)
        log_transformed = True

    if geneset is not None:
        logger.info("Using given geneset")
        interc = adata.var_names.isin(geneset)
        count_data = _get_adata_count_data(adata, layer_key, interc)
        if not log_transformed:
            sc.pp.log1p(adata)
        if adata.raw is None:
            adata.raw = adata
        adata = adata[:, interc]
    else:
        count_data = _get_adata_count_data(adata, layer_key)
        if hvg_method == "seurat_v3":
            try:
                logger.info("Trying seurat_v3 for hvgs")
                sc.pp.highly_variable_genes(
                    adata,
                    flavor="seurat_v3",
                    n_top_genes=n_top_genes,
                    batch_key=batch_key,
                    layer=layer_key,
                )
            except Exception:
                logger.info("Failed, trying pearson residuals for hvgs")
                sc.experimental.pp.highly_variable_genes(
                    adata,
                    n_top_genes=n_top_genes,
                    batch_key=batch_key,
                    layer=layer_key,
                )
        else:
            logger.info("Trying pearson residuals for hvgs")
            sc.experimental.pp.highly_variable_genes(
                adata,
                n_top_genes=n_top_genes,
                batch_key=batch_key,
                layer=layer_key,
            )
        if not log_transformed:
            logger.info("not log_transformed")
            sc.pp.log1p(adata)
        if adata.raw is None:
            adata.raw = adata
        count_data = count_data[:, adata.var.highly_variable]
        adata = adata[:, adata.var.highly_variable]

    if logarithm_after_hvgs:
        logger.info("Log-transform count data")
        count_data = np.log1p(count_data)

    sc.pp.filter_cells(adata, min_genes=0)
    cells = adata.obs["n_genes"] > 10
    adata = adata[cells]
    count_data = count_data[cells]

    count_data = torch.from_numpy(count_data).to(torch.float32)
    batch_label = torch.ones(adata.n_obs)
    batch_df = None
    if layer_key is None:
        logger.info("Adding count data to layer 'counts'")
        layer_key = "counts"
        try:
            adata.layers[layer_key] = count_data.numpy()
        except Exception:
            logger.info("Adding data to 'counts' failed")
    if batch_key is not None:
        catted = adata.obs[batch_key].astype("category")
        batch_df = pd.concat([catted, catted.cat.codes], axis=1)
        batch_df = batch_df.groupby(batch_key).mean()[0].astype(int)
        batch_label = torch.from_numpy(catted.cat.codes.values).long()
    class_label = torch.ones(adata.n_obs)
    class_df = None
    if class_key is not None:
        catted = adata.obs[class_key].astype("category")
        class_df = pd.concat([catted, catted.cat.codes], axis=1)
        class_df = class_df.groupby(class_key).mean()[0].astype(int)
        class_label = torch.from_numpy(catted.cat.codes.values).long()

    return dict(
        gene_expr=count_data,
        adata=adata,
        batch_label=batch_label,
        class_label=class_label,
        _batch_codes=batch_df,
        _class_codes=class_df,
        layer_key=layer_key,
    )


def _get_adata_count_data(adata: AnnData, layer_key=None, interc=None) -> np.ndarray:
    """
    Get the count data from the AnnData object.

    Args:
        adata (AnnData): The input AnnData object.
        layer_key (str, optional): The key for the layer in the AnnData object. Defaults to None.
        interc (np.ndarray, optional): The indices of the genes to be used. Defaults to None.

    Returns:
        np.ndarray: The count data.

    """
    adata = adata[:, interc] if interc is not None else adata
    if layer_key is not None:
        logger.info("Checking layer key")
        count_data = adata.layers[layer_key]
    else:
        count_data = adata.X.copy()  # type:ignore

    if not isinstance(count_data, np.ndarray):
        count_data = count_data.toarray().copy()
    else:
        if (count_data < 0).any():
            try:
                logger.info("Trying .raw")
                logger.info(interc is None)
                adata_raw = adata.raw[:,
                                      interc] if interc is not None else adata.raw
                adata.layers["raw"] = adata_raw.X  # type:ignore
                return _get_adata_count_data(adata, layer_key="raw")
            except Exception:
                raise
    return count_data


def _read_st(adata: Union[str, AnnData]):
    if isinstance(adata, AnnData):
        return adata
    elif os.path.isdir(adata):
        return sc.read_visium(adata)
    return sc.read(adata)


def _read_sc(adata: Union[str, AnnData]):
    if isinstance(adata, AnnData):
        return adata
    elif os.path.isdir(adata):
        return sc.read_10x_h5(adata)
    return sc.read(adata)


def _merge_sc_st_adata(
    sc_adata: AnnData,
    st_adata: AnnData,
    st_sample_names: Optional[Tuple[str, ...]] = None,
    batch_key: Optional[str] = None,
    st_batch_key: Optional[str] = None,
) -> Tuple[AnnData, Tuple[str, ...], str]:
    """
    Merge the single-cell and spatial transcriptomics AnnData objects.

    Args:
        sc_adata (AnnData): The single-cell AnnData object.
        st_adata (AnnData): The spatial transcriptomics AnnData object.
        st_sample_names (Optional[Tuple[str, ...]], optional): The sample names in the spatial transcriptomics data. Defaults to None.
        batch_key (Optional[str], optional): The key for the batch information in the single-cell data. Defaults to None.
        st_batch_key (Optional[str], optional): The key for the batch information in the spatial transcriptomics data. Defaults to None.

    Returns:
        AnnData: The merged AnnData object.
        Tuple[str, ...]: The sample names in the spatial transcriptomics data.
        str: The key for the batch information in the single-cell data.

    """
    if st_batch_key is not None:
        assert (
            st_batch_key in st_adata.obs_keys()
        ), f'`st_batch_key`: "{st_batch_key}" not found in `st_adata.obs`'
        logger.info(f"`st_batch_key` {st_batch_key} found")
        st_adata.obs[st_batch_key] = st_adata.obs[st_batch_key].astype(
            "category")
        st_sample_names = st_adata.obs[st_batch_key].cat.categories

    if st_sample_names is None and st_batch_key is None:
        if "spatial" in st_adata.uns_keys():
            st_sample_names = tuple(
                [sample for sample in st_adata.uns["spatial"].keys()]
            )
            logger.info(f"`st sample name(s)` {st_sample_names} found")
        else:
            logger.warning(
                'sample name(s) of st data is not provided, naming all as "st"'
            )
            st_sample_names = tuple(["st"])

    if len(st_sample_names) > 1:
        logger.info("Multiple st samples found")
        assert st_batch_key is not None, (
            "Must specify `st_batch_key` with multiple st samples scenario"
            + f"Detected samples {st_sample_names} in `.uns['spatial']`"
        )
    else:
        logger.info("Single st samples found")
        if st_batch_key is None:
            st_batch_key = "batch_st"
            st_adata.obs[st_batch_key] = st_sample_names[0]

    if batch_key is None:
        batch_key = "batch"
        sc_adata.obs[batch_key] = "sc"

    st_rename_map = {}
    for st_sample_name in st_sample_names:
        if st_sample_name in sc_adata.obs[batch_key].unique():
            st_rename_map[st_sample_name] = f"{st_sample_name}-ST"

    if batch_key == st_batch_key:
        logger.info("Found conflict between names of sc batch key and st batch key")
        logger.info(f"Rename st batch key {st_batch_key} to {st_batch_key}_orig")
        st_batch_key = f"{st_batch_key}_orig"
        if st_batch_key not in st_adata.obs_keys():
            st_adata.obs.rename(columns={batch_key: st_batch_key}, inplace=True)
    st_adata.obs[batch_key] = st_adata.obs[st_batch_key]

    if st_rename_map:
        logger.info(
            f"Found following conflicted batch name(s) between sc and st: \n{list(st_rename_map.keys())}"
        )
        logger.info(f"Renaming to \n{list(st_rename_map.values())}")
        st_adata.obs[batch_key] = st_adata.obs[st_batch_key].map(st_rename_map)

    sc_adata.obs["modality"] = "sc"
    st_adata.obs["modality"] = "st"
    adata = sc_adata.concatenate(
        st_adata,
        batch_key="modality",
        batch_categories=["SC", "ST"],
        uns_merge="unique",
    )
    return adata, st_sample_names, batch_key


def _filter_mt(adata, is_human):
    mt = "MT" if is_human else "mt"
    adata.var[f"{mt}_gene"] = [gene.startswith(
        f"{mt}-") for gene in adata.var_names]
    # type:ignore
    adata.obsm[mt] = adata[:, adata.var[f"{mt}_gene"].values].X.toarray()
    adata = adata[:, ~adata.var[f"{mt}_gene"].values]  # type:ignore
    return adata


def _assert_dtype(arr):
    assert np.issubdtype(arr.dtype, np.integer) or np.issubdtype(
        arr.dtype, float
    ), "Array dtype is not float or int"


def _ensure_spatial_dtype(adata):
    if "spaital" in adata.obsm_keys() and adata.obsm["spatial"].shape[1] == 2:
        if adata.obsm["spatial"].dtype != float:
            adata.obsm["spatial"] = adata.obsm["spatial"].astype(float)
