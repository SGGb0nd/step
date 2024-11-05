import json
import math
import os
from pathlib import Path
# import random
from typing import Optional

import dgl
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from anndata import AnnData
from matplotlib.image import imread
# from scipy.spatial.distance import cdist
from sklearn.neighbors import BallTree, KDTree

from step.manager import logger


def set_seed(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    os.environ["PYTHONHASHSEED"] = str(seed)
    dgl.random.seed(seed)
    dgl.seed(seed)


def cosine_sim(x, y):
    if len(x.shape) < 2:
        x = x.reshape(1, -1)
    if len(y.shape) < 2:
        y = y.reshape(1, -1)
    return (
        (x * y).sum(-1)
        / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1) + 1e-6)
    )


def neighbors_ranking(
    adata,
    queries,
    batch_key,
    library_id,
    radius=50,
    spatial_key='spatial',
    indvidual_key='X_rep',
    localized_key='X_smoothed',
    return_score=True,
):
    spots = adata.obsm[spatial_key]
    qdata = adata[queries]
    qspots = qdata.obsm[spatial_key]
    cell_emb = qdata.obsm[indvidual_key]
    local_emb = qdata.obsm[localized_key]
    from sklearn.neighbors import KDTree
    tree = KDTree(spots)

    ind = tree.query_radius(qspots, r=radius)
    sorted_barcodes = []
    if return_score:
        sorted_scores = []
    for i, neighbors in enumerate(ind):
        diff_vec = local_emb[i].reshape(1, -1) - cell_emb[neighbors]
        cosine = cosine_sim(cell_emb[i], diff_vec)
        ranked_cosine_indicies = np.argsort(cosine,)
        sorted_barcodes.append(ranked_cosine_indicies)
        if return_score:
            sorted_scores.append(cosine[ranked_cosine_indicies])

    if return_score:
        return sorted_barcodes, sorted_scores
    return sorted_barcodes


def compute_selfattention(
    transformer_encoder, x, mask, src_key_padding_mask, i_layer, d_model, num_heads
):
    h = F.linear(
        x,
        transformer_encoder.layers[i_layer].self_attn.in_proj_weight,
        bias=transformer_encoder.layers[i_layer].self_attn.in_proj_bias,
    )
    qkv = h.reshape(x.shape[0], x.shape[1], num_heads,
                    3 * d_model // num_heads)
    qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
    # [Batch, Head, SeqLen, d_head=d_model//num_heads]
    q, k, v = qkv.chunk(3, dim=-1)
    # [Batch, Head, SeqLen, SeqLen]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    d_k = q.size()[-1]
    attn_probs = attn_logits / math.sqrt(d_k)
    # combining src_mask e.g. upper triangular with src_key_padding_mask e.g. columns over each padding position
    combined_mask = torch.zeros_like(attn_probs)
    if mask is not None:
        combined_mask += mask.float()  # assume mask of shape (seq_len,seq_len)
    if src_key_padding_mask is not None:
        combined_mask += (
            src_key_padding_mask.float()
            .unsqueeze(1)
            .unsqueeze(1)
            .repeat(1, num_heads, x.shape[1], 1)
        )
        # assume shape (batch_size,seq_len), repeating along head and line dimensions == "column" mask
    combined_mask = torch.where(
        combined_mask > 0,
        torch.zeros_like(combined_mask) - float("inf"),
        torch.zeros_like(combined_mask),
    )
    # setting masked logits to -inf before softmax
    attn_probs += combined_mask
    attn_probs = F.softmax(attn_probs, dim=-1)
    return attn_logits, attn_probs


def extract_selfattention_maps(transformer_encoder, x, mask, src_key_padding_mask):
    attn_logits_maps = []
    attn_probs_maps = []
    num_layers = transformer_encoder.num_layers
    d_model = transformer_encoder.layers[0].self_attn.embed_dim
    num_heads = transformer_encoder.layers[0].self_attn.num_heads
    norm_first = transformer_encoder.layers[0].norm_first
    with torch.no_grad():
        for i in range(num_layers):
            # compute attention of layer i
            h = x.clone()
            if norm_first:
                h = transformer_encoder.layers[i].norm1(h)
            # attn = transformer_encoder.layers[i].self_attn(h, h, h,attn_mask=mask,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
            # attention_maps.append(attn) # of shape [batch_size,seq_len,seq_len]
            attn_logits, attn_probs = compute_selfattention(
                transformer_encoder,
                h,
                mask,
                src_key_padding_mask,
                i,
                d_model,
                num_heads,
            )
            attn_logits_maps.append(
                attn_logits
            )  # of shape [batch_size,num_heads,seq_len,seq_len]
            attn_probs_maps.append(attn_probs)
            # forward of layer i
            x = transformer_encoder.layers[i](
                x, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
    return attn_logits_maps, attn_probs_maps


def generate_adj(
    adata, edge_clip: None | float = 2, max_neighbors=6, batch_key=None
) -> dgl.DGLHeteroGraph:
    """
    Generate a graph from spatial coordinates.

    Args:
        adata (anndata.AnnData): Annotated data matrix.
        edge_clip (int): Clip edges at this distance.
        max_neighbors (int): Maximum number of neighbors.
        batch_key (str): Key in adata.obs containing batch information.

    Returns:
        dgl.DGLHeteroGraph: Graph.
    """
    if batch_key is None:
        return _generate_adj(
            adata,
            edge_clip=edge_clip,
            max_neighbors=max_neighbors,
            info=True,
        )
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    graphs = []
    for i, batch in enumerate(adata.obs[batch_key].cat.categories):
        _adata = adata[adata.obs[batch_key] == batch]
        graphs.append(
            _generate_adj(
                _adata, edge_clip=edge_clip, max_neighbors=max_neighbors, info=(i == 0)
            )
        )
    return dgl.batch(graphs=graphs)


def _generate_adj(
    adata, edge_clip: None | float = 2, max_neighbors=6, info=False
) -> dgl.DGLHeteroGraph:
    assert "array_row" in adata.obs.columns
    assert "array_col" in adata.obs.columns
    if edge_clip is None:
        if info:
            logger.info(
                f"Constructing neighbor graph via kNN style: k = {max_neighbors}")
        spots = adata.obs[["array_row", "array_col"]].values
        spots = torch.from_numpy(spots).float()
        # from sklearn.neighbors import KDTree
        tree = KDTree(spots)
        ind = tree.query(spots, k=max_neighbors, return_distance=False)
        g = kdtree_res2adj(ind)
        g.ndata['xy'] = spots
        return g
    if info:
        logger.info(
            f"Constructing neighboring graph via edge clip: clip = {edge_clip}")
    grids = adata.obs[["array_row", "array_col"]].values.astype(np.float32)
    # try:
    #     m = cdist(grids, grids)
    #     m[m <= edge_clip] = 1
    #     m[m > edge_clip] = 0
    #     m = sp.csr_matrix(m)
    # except Exception:
    if edge_clip == 'visium':
        edge_clip = 2
        tree = BallTree(grids, metric=visium_grid_distance)
    else:
        tree = KDTree(grids,)

    ind = tree.query_radius(grids, r=edge_clip)
    g = kdtree_res2adj(ind)
    g.ndata['xy'] = torch.from_numpy(grids).float()
    return g


def kdtree_res2adj(ind):
    rows = np.concatenate([np.full(len(neighbors), i) for i, neighbors in enumerate(ind)])
    cols = np.concatenate(ind)
    n_points = len(ind)

    m = sp.csr_matrix((np.ones_like(rows), (rows, cols)), shape=(n_points, n_points))
    g = dgl.from_scipy(m)
    g = g.remove_self_loop()
    g = dgl.add_self_loop(g)
    return g


def visium_grid_distance(x, y):
    if abs(y[1] - x[1]) >= 2:
        return np.inf
    return np.linalg.norm(x - y)


def z2polar(edges):
    z = edges.dst["xy"] - edges.src["xy"]
    rho = torch.norm(z, dim=-1, p=2)
    x, y = z.unbind(dim=-1)
    phi = torch.atan2(y, x)
    return {"polar": torch.cat([rho.unsqueeze(-1), phi.unsqueeze(-1)], -1)}


def z2direction(edges):
    num_directions = 6
    z = edges.dst["xy"] - edges.src["xy"]
    x, y = z.unbind(dim=-1)
    phi = torch.atan2(y, x)
    directions = torch.floor((phi + math.pi) / (2 * math.pi / num_directions)).long() % num_directions
    return {"direction": directions}


def aver_items_by_ct(
    adata: AnnData,
    cell_type_key: str,
    items: torch.Tensor | np.ndarray,
    weight: Optional[torch.Tensor] = None,
    return_cts=False,
):
    """Generate anchors for each cell type based on the given data and transformation output.

    Args:
        adata (AnnData): Annotated data object.
        cell_type_key (str): Key for the cell type column in the adata object.
        tsfmr_out (torch.Tensor): Transformation output tensor.
        weight (Optional[torch.Tensor]): Weight tensor for aggregation. Default is None.
        return_cts (bool): Whether to return the cell types along with the anchors. Default is False.

    Returns:
        torch.Tensor or Tuple[torch.Tensor, List[str]]: Anchors tensor or tuple of anchors tensor and list of cell types.
    """

    assert cell_type_key in adata.obs_keys()
    anchors = []
    if weight is None:
        def agg_func(x, i, mask):
            return x[mask].mean(0)
    else:
        agg_func = (
            lambda x, i, mask: weight[mask, i]
            / (weight[mask, i].sum()) @ x[mask]
        )
    celltypes = adata.obs[cell_type_key].astype("category").cat.categories
    for i, ct in enumerate(celltypes):
        mask = (adata.obs[cell_type_key] == ct).values
        anchors.append(agg_func(items, i, mask))
    if isinstance(anchors[0], np.ndarray):
        anchors = np.stack(anchors)
    else:
        anchors = torch.stack(anchors)
    if return_cts:
        return anchors, list(celltypes)
    return anchors


def read_visium_hd(
    path: Path | str,
    genome: str | None = None,
    *,
    count_file: str = "filtered_feature_bc_matrix.h5",
    library_id: str | None = None,
    load_images: bool | None = True,
    source_image_path: Path | str | None = None,
) -> AnnData:
    """\
    Read 10x-Genomics-formatted visum-hd dataset, **modified from scanpy.read_visium**.

    In addition to reading regular 10x output,
    this looks for the `spatial` folder and loads images,
    coordinates and scale factors.
    Based on the `Space Ranger output docs`_.

    See :func:`~scanpy.pl.spatial` for a compatible plotting function.

    .. _Space Ranger output docs: https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview

    Parameters
    ----------
    path
        Path to directory for visium datafiles.
    genome
        Filter expression to genes within this genome.
    count_file
        Which file in the passed directory to use as the count file. Typically would be one of:
        'filtered_feature_bc_matrix.h5' or 'raw_feature_bc_matrix.h5'.
    library_id
        Identifier for the visium library. Can be modified when concatenating multiple adata objects.
    source_image_path
        Path to the high-resolution tissue image. Path will be included in
        `.uns["spatial"][library_id]["metadata"]["source_image_path"]`.

    Returns
    -------
    Annotated data matrix, where observations/cells are named by their
    barcode and variables/genes by gene name. Stores the following information:

    :attr:`~anndata.AnnData.X`
        The data matrix is stored
    :attr:`~anndata.AnnData.obs_names`
        Cell names
    :attr:`~anndata.AnnData.var_names`
        Gene names for a feature barcode matrix, probe names for a probe bc matrix
    :attr:`~anndata.AnnData.var`\\ `['gene_ids']`
        Gene IDs
    :attr:`~anndata.AnnData.var`\\ `['feature_types']`
        Feature types
    :attr:`~anndata.AnnData.obs`\\ `[filtered_barcodes]`
        filtered barcodes if present in the matrix
    :attr:`~anndata.AnnData.var`
        Any additional metadata present in /matrix/features is read in.
    :attr:`~anndata.AnnData.uns`\\ `['spatial']`
        Dict of spaceranger output files with 'library_id' as key
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['images']`
        Dict of images (`'hires'` and `'lowres'`)
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['scalefactors']`
        Scale factors for the spots
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['metadata']`
        Files metadata: 'chemistry_description', 'software_version', 'source_image_path'
    :attr:`~anndata.AnnData.obsm`\\ `['spatial']`
        Spatial spot coordinates, usable as `basis` by :func:`~scanpy.pl.embedding`.
    """
    path = Path(path)
    adata = sc.read_10x_h5(path / count_file, genome=genome)

    adata.uns["spatial"] = dict()

    from h5py import File

    with File(path / count_file, mode="r") as f:
        attrs = dict(f.attrs)
    if library_id is None:
        library_id = str(attrs.pop("library_ids")[0], "utf-8")

    adata.uns["spatial"][library_id] = dict()

    if load_images:
        tissue_positions_file = (
            path / "spatial/tissue_positions.parquet"
            if (path / "spatial/tissue_positions.parquet").exists()
            else path / "spatial/tissue_positions_list.parquet"
        )
        files = dict(
            tissue_positions_file=tissue_positions_file,
            scalefactors_json_file=path / "spatial/scalefactors_json.json",
            hires_image=path / "spatial/tissue_hires_image.png",
            lowres_image=path / "spatial/tissue_lowres_image.png",
        )

        # check if files exists, continue if images are missing
        for f in files.values():
            if not f.exists():
                if any(x in str(f) for x in ["hires_image", "lowres_image"]):
                    logger.warning(
                        f"You seem to be missing an image file.\n"
                        f"Could not find '{f}'."
                    )
                else:
                    raise OSError(f"Could not find '{f}'")

        adata.uns["spatial"][library_id]["images"] = dict()
        for res in ["hires", "lowres"]:
            try:
                adata.uns["spatial"][library_id]["images"][res] = imread(
                    str(files[f"{res}_image"])
                )
            except Exception:
                raise OSError(f"Could not find '{res}_image'")

        # read json scalefactors
        adata.uns["spatial"][library_id]["scalefactors"] = json.loads(
            files["scalefactors_json_file"].read_bytes()
        )

        adata.uns["spatial"][library_id]["metadata"] = {
            k: (str(attrs[k], "utf-8") if isinstance(attrs[k], bytes) else attrs[k])
            for k in ("chemistry_description", "software_version")
            if k in attrs
        }

        # read coordinates
        positions = pd.read_parquet(
            files["tissue_positions_file"],
        )
        positions.set_index("barcode", inplace=True)
        positions.columns = [
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_col_in_fullres",
            "pxl_row_in_fullres",
        ]

        adata.obs = adata.obs.join(positions, how="left")

        adata.obsm["spatial"] = adata.obs[
            ["pxl_row_in_fullres", "pxl_col_in_fullres"]
        ].to_numpy()
        adata.obs.drop(
            columns=["pxl_row_in_fullres", "pxl_col_in_fullres"],
            inplace=True,
        )

        # put image path in uns
        if source_image_path is not None:
            # get an absolute path
            source_image_path = str(Path(source_image_path).resolve())
            adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = str(
                source_image_path
            )

    return adata
