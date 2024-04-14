from typing import Optional

import numpy as np
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors

from step.manager import logger


def knn_random_coords(
    adata: AnnData,
    Y_spatial: np.ndarray,
    k: int = 6,
    use_rep: str = "X_rep",
    library_id: Optional[str] = None,
):
    """Assigns random coordinates to the nearest neighbors from scRNA-seq data for spatial data.

    Args:
        adata (AnnData): Annotated data object.
        Y_spatial (np.ndarray): Spatial coordinates.
        k (int, optional): Number of nearest neighbors. Defaults to 6.
        use_rep (str, optional): Representation to use. Defaults to 'X_rep'.
        library_id (str, optional): Library ID. Defaults to None.

    Returns:
        AnnData: AnnData object of spatial data with scRNA-seq data mapped.
    """
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    X = adata[adata.obs["modality"] == "SC"].obsm[use_rep]
    Y = adata[adata.obs["modality"] == "ST"].obsm[use_rep]

    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    indices = nbrs.kneighbors(Y, return_distance=False)
    try:
        radius = adata.uns["spatial"][library_id]["scalefactors"][
            "spot_diameter_fullres"
        ]
    except Exception:
        logger.info(
            "Cannot find spot size in `adata.uns`. Using mini distance as diameter instead"
        )
        from scipy.spatial.distance import cdist

        mat = cdist(Y_spatial, Y_spatial)
        radius = mat[mat > 0].min() / 2

    random_offsets = np.random.uniform(-radius,
                                       radius, size=(*indices.shape, 2))

    reshape_Y_spatial = Y_spatial.reshape((-1, 1, Y_spatial.shape[1]))
    assigned_coords = (reshape_Y_spatial + random_offsets).reshape(-1, 2)

    SC_indices = (
        adata[adata.obs["modality"]
              == "SC"].obs.iloc[indices.reshape(-1)].index
    )

    knn_adata = adata[SC_indices.append(
        adata[adata.obs["modality"] == "ST"].obs_names)]
    knn_adata.obsm["spatial"] = np.zeros((knn_adata.n_obs, 2))

    knn_adata.obsm["spatial"][knn_adata.obs["modality"]
                              == "SC"] = assigned_coords
    knn_adata.obsm["spatial"][knn_adata.obs["modality"] == "ST"] = Y_spatial
    return knn_adata
