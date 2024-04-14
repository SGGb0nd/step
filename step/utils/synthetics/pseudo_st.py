import os
from typing import Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from torch.distributions import Normal
from tqdm import tqdm

from step.manager import logger
from step.utils.misc import generate_adj


class PseudoST:
    """
    Pseudo spatial transcriptomics data.

    Attributes:
        adata_sc (AnnData): Single cell data.
        adata_st (AnnData): Spatial transcriptomics data.
        layer_key (str): Layer key for single cell data.
        batch_key (str): Batch key for single cell data.
        k_hop (int): Number of hops for spatial smoothing.
        downsapmle_size (int): Downsample size for single cell data.
        cell_type_key (str): Cell type key for single cell data.
        sub_type_key (str): Sub type key for single cell data.
        coords_key (str): Coords key for spatial transcriptomics data.
        info_key (str): Info key for spatial transcriptomics data.

    """

    def __init__(
        self,
        adata_sc: AnnData,
        adata_st: AnnData,
        layer_key="counts",
        batch_key=None,
        k_hop=3,
        downsapmle_size=200,
        cell_type_key="cell_type",
        sub_type_key=None,
        coords_key="spatial",
        info_key="spatial",
    ):
        """
        Initialize PseudoST.

        Args:
            adata_sc (AnnData): Single cell data.
            adata_st (AnnData): Spatial transcriptomics data.
            layer_key (str): Layer key for single cell data.
            batch_key (str): Batch key for single cell data.
            k_hop (int): Number of hops for spatial smoothing.
            downsapmle_size (int): Downsample size for single cell data.
            cell_type_key (str): Cell type key for single cell data.
            sub_type_key (str): Sub type key for single cell data.
            coords_key (str): Coords key for spatial transcriptomics data.
            info_key (str): Info key for spatial transcriptomics data.
        """
        if layer_key is None:
            adata_sc.layers["counts"] = adata_sc.X
            layer_key = "counts"
        assert (
            layer_key in adata_sc.layers.keys()
        ), f"layer_key {layer_key} not in adata_sc"
        assert (
            cell_type_key in adata_sc.obs.columns
        ), f"cell_type_key {cell_type_key} not in adata_sc"
        adata_sc = adata_sc[adata_sc.obs[cell_type_key].notna()]
        self.adata_sc = adata_sc
        self.ct_key = cell_type_key
        self.k_hop = k_hop
        self._downsample_sc_data(batch_key=batch_key, size=downsapmle_size)
        cts = self.adata_sc.obs[cell_type_key].astype(
            "category").cat.categories
        self.cts = cts

        self.layer_key = layer_key
        self._coords_key = coords_key
        self._info_key = info_key
        self.adata_st = self._setup_st_data(adata_st)

        self._domain_num_major_map = {}
        self._domain_ct_map = {}
        if sub_type_key is not None:
            sub_cts = self.adata_sc.obs[sub_type_key].astype(
                "category").cat.categories
            tmp_df = self.adata_sc.obs[
                [
                    sub_type_key,
                    cell_type_key,
                ]
            ].apply(lambda x: x.cat.codes)
            self.sub_cts = sub_cts
            self.sub_ct_key = sub_type_key
            self.ct_ref = (
                tmp_df.groupby(cell_type_key)[sub_type_key]
                .apply(lambda x: x.unique())
                .to_dict()
            )

            self.domain_ref = {}
            self._subdomain_num_major_map = {}
            self._subdomain_ct_map = {}

    def _valid_st_data(self, adata_st: AnnData):
        """
        Validate spatial transcriptomics data.

        Args:
            adata_st (AnnData): Spatial transcriptomics data.

        Raises:
            KeyError: If coords_key or info_key not in adata_st.
        """
        try:
            adata_st.obsm[self._coords_key]
        except KeyError:
            raise
        try:
            adata_st.uns[self._info_key]
        except KeyError:
            raise

    def _setup_st_data(self, adata_st: AnnData):
        """
        Setup simulated spatial transcriptomics data.

        Args:
            adata_st (AnnData): Spatial transcriptomics data.

        Returns:
            AnnData: Simulated spatial transcriptomics data.
        """
        self._valid_st_data(adata_st)
        pseudo_adata_st = AnnData(
            X=np.zeros((adata_st.n_obs, self.adata_sc.n_vars)),
            obs=adata_st.obs.copy(),
            var=self.adata_sc.var.copy(),
            uns=adata_st.uns,
        )
        sample_name = list(adata_st.uns["spatial"].keys())[0]
        adata_st.uns["spatial"][f"SIMU-{sample_name}"] = adata_st.uns["spatial"].pop(
            sample_name
        )
        pseudo_adata_st.obsm[self._coords_key] = adata_st.obsm[self._coords_key]
        del pseudo_adata_st.var
        return pseudo_adata_st

    def _downsample_sc_data(self, batch_key=None, size=100):
        """
        Downsample single cell data.

        Args:
            batch_key (str): Batch key for single cell data.
            size (int): Downsample size for single cell data.
        """
        # if single batch, just sample from adata_sc per cell type
        downsampled = []
        if batch_key is None:
            all_ind = np.arange(len(self.adata_sc))
            for ct in self.cts:
                _ct_indicies = all_ind[self.adata_sc.obs[self.ct_key] == ct]
                indicies = np.random.choice(_ct_indicies, size=size)
                downsampled.extend(indicies)
        else:
            # random choose a batch, then only use the cell types in that batch
            batches = self.adata_sc.obs[batch_key].unique()
            batch = np.random.choice(batches)
            logger.info(f"Using batch: {batch} with downsample_size: {size}")
            indicies = self.adata_sc.obs[batch_key] == batch
            self.adata_sc = self.adata_sc[indicies]
            self.cts = self.adata_sc.obs[self.ct_key].astype(
                "category").cat.categories
            all_ind = np.arange(len(self.adata_sc))

            for ct in self.cts:
                _ct_indicies = all_ind[self.adata_sc.obs[self.ct_key] == ct]
                indicies = np.random.choice(_ct_indicies, size=size)
                downsampled.extend(indicies)

        self.adata_sc = self.adata_sc[downsampled]

    def _init_num_major_types(
        self,
        domain_key,
        max_num_major_types=2,
        sub_domain_key=None,
        max_num_major_subtypes=2,
    ):
        """
        Initialize number of major cell types.

        Args:
            domain_key (str): Domain key for spatial transcriptomics data.
            max_num_major_types (int): Maximum number of major cell types.
            sub_domain_key (str): Sub domain key for spatial transcriptomics data.
            max_num_major_subtypes (int): Maximum number of major sub cell types.

        Raises:
            ValueError: If domain_key not in adata_st.
        """
        for domain in self.adata_st.obs[domain_key].cat.codes:
            self._domain_num_major_map[domain] = np.random.randint(
                1, max_num_major_types + 1
            )
        if sub_domain_key is not None:
            for domain in self.adata_st.obs[sub_domain_key].cat.codes:
                self._subdomain_num_major_map[domain] = np.random.randint(
                    2, max_num_major_subtypes + 1
                )
            self.domain_ref = dict(
                self.adata_st.obs[
                    [
                        domain_key,
                        sub_domain_key,
                    ]
                ]
                .apply(lambda x: x.cat.codes)
                .value_counts()
                .sort_index()
                .swaplevel(0, 1)
                .index
            )

    def _assign_major_types(
        self,
        domain_key,
        mean_prop=0.7,
        max_num_major_types=2,
        sub_domain_key=None,
        max_num_major_subtypes=4,
    ):
        """
        Assign major cell types to each domain.

        Args:
            domain_key (str): Domain key for spatial transcriptomics data.
            mean_prop (float): Mean proportion of major cell types.
            max_num_major_types (int): Maximum number of major cell types.
            sub_domain_key (str): Sub domain key for spatial transcriptomics data.
            max_num_major_subtypes (int): Maximum number of major sub cell types.

        Raises:
            ValueError: If domain_key not in adata_st.
        """
        domain_ct_map = {}
        sub_domain_ct_map = {}
        cell_types = self.cts
        sub_cell_types = self.sub_cts
        self._init_num_major_types(
            domain_key,
            max_num_major_types=max_num_major_types,
            sub_domain_key=sub_domain_key,
            max_num_major_subtypes=max_num_major_subtypes,
        )
        num_cts = len(self.cts)
        sample_p = np.array([1 / num_cts] * num_cts)
        for domain, num_major_types in self._domain_num_major_map.items():
            domain_ct_map[domain] = {}
            major_types = np.random.choice(
                len(cell_types), size=num_major_types, p=sample_p, replace=False
            ).astype(int)
            domain_ct_map[domain]["major_cts"] = major_types
            # lower the sample probability for the major types
            sample_p[major_types] = sample_p[major_types] / 2
            sample_p = sample_p / sample_p.sum()

            if num_major_types > 1:
                min_prop = 1 - mean_prop
                max_prop = mean_prop - min_prop
                p = np.random.uniform(min_prop, max_prop, num_major_types)
                p = p / p.sum() * mean_prop
                domain_ct_map[domain]["major_props"] = p
            else:
                domain_ct_map[domain]["major_props"] = np.array([mean_prop])
        self._domain_ct_map = domain_ct_map

        if sub_cell_types is not None:
            for domain, num_major_types in self._subdomain_num_major_map.items():
                sub_domain_ct_map[domain] = {}
                sup_domain = self.domain_ref[domain]
                sup_cts = self._domain_ct_map[sup_domain]["major_cts"]
                sup_props = self._domain_ct_map[sup_domain]["major_props"]

                sub_domain_ct_map[domain]["major_cts"] = []
                sub_domain_ct_map[domain]["major_props"] = []
                for i, ct in enumerate(sup_cts):
                    choices = self.ct_ref[ct]
                    num_sub_cts = len(choices)
                    sample_p = np.array([1 / num_sub_cts] * num_sub_cts)
                    major_types_ind = np.random.choice(
                        num_sub_cts,
                        size=min(num_major_types, num_sub_cts),
                        p=sample_p,
                        replace=False,
                    ).astype(int)
                    major_types_ind = np.unique(major_types_ind)
                    major_types = choices[major_types_ind]
                    sub_domain_ct_map[domain]["major_cts"].append(major_types)
                    # lower the sample probability for the major types
                    sample_p[major_types_ind] = sample_p[major_types_ind] / 2
                    sample_p = sample_p / sample_p.sum()

                    min_prop = 1 - mean_prop
                    max_prop = mean_prop - min_prop
                    p = np.random.uniform(min_prop, max_prop, len(major_types))
                    p = p / p.sum() * sup_props[i]
                    sub_domain_ct_map[domain]["major_props"].append(p)

                sub_domain_ct_map[domain]["major_cts"] = np.concatenate(
                    sub_domain_ct_map[domain]["major_cts"]
                )
                sub_domain_ct_map[domain]["major_props"] = np.concatenate(
                    sub_domain_ct_map[domain]["major_props"]
                )
            self._subdomain_ct_map = sub_domain_ct_map

    def _assign_minor_types(self, max_rank=6):
        """
        Assign minor cell types to each domain.

        Args:
            max_rank (int): Maximum rank for minor cell types.

        """
        domain_ct_map = self._domain_ct_map
        cell_types = self.cts
        for domain, ct_info in domain_ct_map.items():
            num_major_types = len(ct_info["major_cts"])
            minor_types = np.random.uniform(
                0, len(cell_types), max_rank - num_major_types
            ).astype(int)
            minor_types = np.setdiff1d(minor_types, ct_info["major_cts"])
            domain_ct_map[domain]["minor_cts"] = minor_types

            num_minor_types = len(minor_types)
            domain_ct_map[domain]["minor_props"] = np.random.dirichlet(
                np.ones(num_minor_types), size=1
            )[0]

        self._domain_ct_map = domain_ct_map

        sub_cell_types = self.sub_cts
        if sub_cell_types is not None:
            sub_domain_ct_map = self._subdomain_ct_map
            for domain, ct_info in sub_domain_ct_map.items():
                sub_domain_ct_map[domain]["minor_cts"] = []
                sub_domain_ct_map[domain]["minor_props"] = []
                sup_domain = self.domain_ref[domain]
                sup_cts = self._domain_ct_map[sup_domain]["minor_cts"]
                for ct in sup_cts:
                    choices = self.ct_ref[ct]
                    num_sub_cts = len(choices)
                    sample_p = np.array([1 / num_sub_cts] * num_sub_cts)
                    minor_types = np.random.choice(
                        choices,
                        size=1,
                        p=sample_p,
                    ).astype(int)
                    sub_domain_ct_map[domain]["minor_cts"].extend(minor_types)

                    num_minor_types = len(minor_types)
                    sub_domain_ct_map[domain]["minor_props"].extend(
                        np.random.uniform(size=num_minor_types)
                    )
                sub_domain_ct_map[domain]["minor_props"] = np.asarray(
                    sub_domain_ct_map[domain]["minor_props"]
                )
                sub_domain_ct_map[domain]["minor_props"] /= sub_domain_ct_map[domain][
                    "minor_props"
                ].sum()

        self._subdomain_ct_map = sub_domain_ct_map

    def _sample_ct_composition(self, domain_key, use_sub=False):
        """
        Sample cell type composition.

        Args:
            domain_key (str): Domain key for spatial transcriptomics data.
            use_sub (bool): Use sub domain key.

        Returns:
            np.ndarray: Cell type composition.
        """
        domain_sampler = {}
        if not use_sub:
            domain_ct_map = self._domain_ct_map
            ct_key = self.ct_key
            scale = 0.1
        else:
            domain_ct_map = self._subdomain_ct_map
            ct_key = self.sub_ct_key
            scale = 0.05

        num_domains = len(domain_ct_map)
        cell_types = self.adata_sc.obs[ct_key].unique()
        num_cts = len(cell_types)
        props = np.zeros((num_domains, num_cts))
        for domain, ct_info in domain_ct_map.items():
            props[domain, ct_info["major_cts"]] = ct_info["major_props"]
            props[domain, ct_info["minor_cts"]] = ct_info["minor_props"] * (
                1 - ct_info["major_props"].sum()
            )
            loc = torch.from_numpy(props[domain]).float()
            domain_sampler[domain] = Normal(loc=loc, scale=scale)

        del props
        spot_ct_composition = np.stack(
            [
                domain_sampler[domain].sample().numpy()
                for domain in self.adata_st.obs[domain_key].cat.codes
            ]
        )
        spot_ct_composition[spot_ct_composition < 0] = 0
        spot_ct_composition = spot_ct_composition / spot_ct_composition.sum(
            axis=1, keepdims=True
        )
        return spot_ct_composition

    def _spatial_smoothing(self, spot_ct_composition):
        """
        Spatial smoothing.

        Args:
            spot_ct_composition (np.ndarray): Spot cell type composition.

        Returns:
            np.ndarray: Smoothed spot cell type composition.
        """
        dgl_adj = generate_adj(self.adata_st)
        adj = dgl_adj.adjacency_matrix().to_dense().numpy()
        if isinstance(self.k_hop, int) and self.k_hop > 0:
            smoothed_spot_ct_composition_ = (
                adj**self.k_hop) @ spot_ct_composition
        else:
            smoothed_spot_ct_composition_ = spot_ct_composition
        smoothed_spot_ct_composition = (
            smoothed_spot_ct_composition_
            / smoothed_spot_ct_composition_.sum(axis=1, keepdims=True)
        )
        return smoothed_spot_ct_composition

    def _sample_single_cell_mix(
        self, spot_ct_comp, min_cells=2, max_cells=7, use_sub=False
    ):
        """
        Sample single cell mix.

        Args:
            spot_ct_comp (np.ndarray): Spot cell type composition.
            min_cells (int): Minimum number of cells.
            max_cells (int): Maximum number of cells.
            use_sub (bool): Use sub domain key.

        Returns:
            np.ndarray: Pseudo counts.
        """
        num_cells = np.random.randint(
            min_cells, max_cells + 1, size=self.adata_st.n_obs
        )
        self.adata_st.obs["num_cells"] = num_cells

        num_cells_per_ct = spot_ct_comp * num_cells[:, None]
        cell_types = self.cts
        ct_key = self.ct_key
        if use_sub:
            cell_types = self.sub_cts
            ct_key = self.sub_ct_key
        pseudo_counts = np.zeros((self.adata_st.n_obs, self.adata_sc.n_vars))
        with tqdm(range(self.adata_st.n_obs)) as tobs:
            for i in tobs:
                signatures = np.zeros((len(cell_types), self.adata_sc.n_vars))
                for j, ct in enumerate(cell_types):
                    if num_cells_per_ct[i, j] == 0:
                        continue
                    _adata: AnnData = self.adata_sc[self.adata_sc.obs[ct_key] == ct]
                    num_ct_cells = round(10.4 + num_cells_per_ct[i, j])
                    try:
                        ct_index = np.random.choice(
                            _adata.n_obs, size=num_ct_cells, replace=True
                        )
                    except Exception:
                        logger.info(ct, _adata.n_obs)
                        raise
                    counts = _adata.layers[self.layer_key]
                    signatures[j] = counts[ct_index].mean(axis=0)
                pseudo_counts[i] = num_cells_per_ct[i] @ signatures

        return pseudo_counts

    def simulate(
        self,
        domain_key: str,
        mean_prop=0.75,
        min_cells: int = 2,
        max_cells: int = 7,
        max_num_major_types: int = 2,
        max_rank: Union[int, float] = 0.5,
        prop_key="ct_composition",
        abund_key="ct_abundance",
        sub_domain_key=None,
        max_num_major_subtypes=4,
    ):
        """
        Simulate spatial transcriptomics data.

        Args:
            domain_key (str): Domain key for spatial transcriptomics data.
            mean_prop (float): Mean proportion of major cell types.
            min_cells (int): Minimum number of cells.
            max_cells (int): Maximum number of cells.
            max_num_major_types (int): Maximum number of major cell types.
            max_rank (Union[int, float]): Maximum rank for minor cell types.
            prop_key (str): Key for cell type composition.
            abund_key (str): Key for cell type abundance.
            sub_domain_key (str): Sub domain key for spatial transcriptomics data.
            max_num_major_subtypes (int): Maximum number of major sub cell types.

        Returns:
            AnnData: Simulated spatial transcriptomics data.
        """
        assert (
            domain_key in self.adata_st.obs.columns
        ), f"domain_key {domain_key} not in st data"
        self._assign_major_types(
            domain_key,
            max_num_major_types=max_num_major_types,
            mean_prop=mean_prop,
            sub_domain_key=sub_domain_key,
            max_num_major_subtypes=max_num_major_subtypes,
        )
        if isinstance(max_rank, float):
            max_rank = int(len(self.cts) * max_rank)
        self._assign_minor_types(max_rank=max_rank)
        if sub_domain_key is not None:
            logger.info(f"Using sub domain key {sub_domain_key}")
            use_domain_key = sub_domain_key
            cts = self.sub_cts
            use_sub = True
        else:
            use_sub = False
            cts = self.cts
        spot_ct_composition = self._sample_ct_composition(
            use_domain_key, use_sub=use_sub
        )
        (smoothed_spot_ct_composition) = self._spatial_smoothing(spot_ct_composition)
        self.adata_st.obsm[prop_key] = pd.DataFrame(
            smoothed_spot_ct_composition,
            index=self.adata_st.obs_names,
            columns=cts,
        )
        self.adata_st.obs[cts.tolist()] = smoothed_spot_ct_composition
        pesudo_counts = self._sample_single_cell_mix(
            smoothed_spot_ct_composition,
            min_cells=min_cells,
            max_cells=max_cells,
            use_sub=True,
        )
        self.adata_st.X = pesudo_counts

        self.adata_st.obsm[abund_key] = pd.DataFrame(
            smoothed_spot_ct_composition
            * self.adata_st.obs["num_cells"].values[:, None],
            index=self.adata_st.obs_names,
            columns=cts,
        )
        return self.adata_st

    def save(self, file, st_only=True, **kwargs):
        """
        Save simulated spatial transcriptomics data.

        Args:
            file (str): File to save.
            st_only (bool): Save only spatial transcriptomics data.
            **kwargs: Additional arguments.

        """
        if st_only:
            self.adata_st.write_h5ad(filename=file, **kwargs)
            return
        st_file = _genertae_path_for_save(file, "st")
        sc_file = _genertae_path_for_save(file, "sc")
        self.adata_st.write_h5ad(st_file, **kwargs)  # type: ignore
        self.adata_sc.write_h5ad(sc_file, **kwargs)  # type: ignore
        return


def _genertae_path_for_save(file, prefix):
    prefix_added_file = file.split(".")
    prefix_added_file[-2] = f"{prefix_added_file[-2]}_{prefix}"
    prefix_added_file = ".".join(prefix_added_file)
    prefix_added_file = os.path.abspath(prefix_added_file)
    return prefix_added_file
