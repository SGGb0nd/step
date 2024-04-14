import esda
import libpysal


def moran_i(adata, coord_keys, obs_key, k=1):
    """
    Calculate Moran's I for a given spatially distributed variable.

    Args:
        adata (anndata.AnnData): Annotated data matrix.
        coord_keys (list): List of keys in adata.obsm containing spatial coordinates.
        obs_key (str): Key in adata.obs containing the variable of interest.
        k (int): Number of nearest neighbors to consider.

    Returns:
        float: Moran's I value.
    """
    coordinates = adata.obsm[coord_keys]
    w_knn = libpysal.weights.KNN.from_array(
        coordinates,
        k=k,
    )
    values = adata.obs[obs_key].cat.codes
    moran = esda.Moran(values, w_knn)
    moran_i = moran.I
    return moran_i


def geary_c(adata, coord_keys, obs_key, k=1):
    """
    Calculate Geary's C for a given spatially distributed variable.

    Args:
        adata (anndata.AnnData): Annotated data matrix.
        coord_keys (list): List of keys in adata.obsm containing spatial coordinates.
        obs_key (str): Key in adata.obs containing the variable of interest.
        k (int): Number of nearest neighbors to consider.

    Returns:
        float: Geary's C value.
    """
    coordinates = adata.obsm[coord_keys]
    w_knn = libpysal.weights.KNN.from_array(
        coordinates,
        k=k,
    )
    values = adata.obs[obs_key].cat.codes
    geary = esda.Geary(values, w_knn)
    geary_c = geary.C
    return geary_c
