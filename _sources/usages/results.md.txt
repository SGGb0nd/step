# Results Storage

## Running artifacts
### Keys of embedding results
1. The embedding results are stored in `adata.obsm['X_rep']` for the `scModel` and `crossModel` objects.
2. The embedding results with maunual annotation refinement are stored in `adata.obsm['X_anchord']` for the `scModel` and `crossModel` objects.
3. The embedding results with spatially smoothed are stored in `adata.obsm['X_smoothed']` for the `stModel` objects.
4. The embedding results without spatially smoothed are stored in `adata.obsm['X_unsmoothed']` for the `stModel` objects (only when not running on e2e mode).

### Keys of spatial domain identification results
1. The spatial domain identification results are stored in `adata.obs['domain']` for the `stModel` and `crossModel` objects.
2. The spatial sub-domain identification results are stored in `adata.obs['sub_domain']` for the `stModel` and `crossModel` objects.

### Keys of deconvolution results
1. The normalized (sum=1) deconvolution results are stored in `adata.obsm['deconv']` for the `crossModel` objects.
2. The un-normalized deconvolution results are stored in `adata.obs['deconv_abundance']` for the `crossModel` objects.

### Keys of batch-corrected and imputed gene expression profiles
1. The batch-corrected gene expression profiles are stored in `adata.layers['corrected_counts']` for the `scModel` and `crossModel` objects.
2. The imputed gene expression profiles are stored in `adata.layers['imputed_counts']` for the `scModel` and `crossModel` objects.

## Saving and loading artifacts
### Saving artifacts
All the artifacts can be saved by the `save` method of the `scModel`, `stModel`, and `crossModel` objects. The saved artifacts include the `adata` object, the model weights, and the model configurations.

### Loading artifacts
All the artifacts can be loaded by the `load` method of the `scModel`, `stModel`, and `crossModel` objects. The loaded artifacts include the `adata` object, the model weights, and the model configurations.