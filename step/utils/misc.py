import math
import os
# import random
from typing import Optional

import dgl
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from anndata import AnnData
from scipy.spatial.distance import cdist

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


def get_de_genes(
    adata,
    cell_type_key,
    de_key="rank_genes_groups_filtered",
    n_genes=10,
    log2fc_cutoff=0.5,
    hvg_intersection=True,
) -> np.ndarray:
    cell_type_names = adata.obs[cell_type_key].cat.categories
    if de_key not in adata.uns:
        sc.tl.rank_genes_groups(
            adata,
            cell_type_key,
            method="wilcoxon",
            use_raw=True,
        )
        sc.tl.filter_rank_genes_groups(
            adata,
            min_fold_change=log2fc_cutoff,
            min_in_group_fraction=0.25,
        )
    de_results = adata.uns[de_key]
    de_genes = []
    for cell_type_name in cell_type_names:
        de_genes.extend(
            de_results["names"][cell_type_name][:n_genes]
        )
    if hvg_intersection:
        return adata.var_names.isin(de_genes).flatten()
    return de_genes


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
    adata, edge_clip: Optional[int] = 2, max_neighbors=6, batch_key=None
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
        )
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    graphs = []
    for i, batch in enumerate(adata.obs[batch_key].cat.categories):
        _adata = adata[adata.obs[batch_key] == batch]
        graphs.append(
            _generate_adj(
                _adata, edge_clip=edge_clip, max_neighbors=max_neighbors, info=i == 0
            )
        )
    return dgl.batch(graphs=graphs)


def _generate_adj(
    adata, edge_clip: Optional[int] = 2, max_neighbors=6, info=False
) -> dgl.DGLHeteroGraph:
    assert "array_row" in adata.obs.columns
    assert "array_col" in adata.obs.columns
    if edge_clip is None:
        if info:
            logger.info(
                f"Constructing neighbor graph via kNN style: k = {max_neighbors}")
        spots = adata.obs[["array_row", "array_col"]].values
        spots = torch.from_numpy(spots).float()
        g = dgl.knn_graph(spots, max_neighbors)
        g = dgl.add_self_loop(g)
        return g
    if info:
        logger.info(
            f"Constructing neighboring graph via edge clip: clip = {edge_clip}")
    grids = adata.obs[["array_row", "array_col"]].values
    m = cdist(grids, grids)
    m[m <= edge_clip] = 1
    m[m > edge_clip] = 0
    g = dgl.from_scipy(sp.csr_matrix(m))
    g = dgl.add_self_loop(g)
    return g


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
