from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData

from step.manager import logger


def spatial_plot(
    adata: AnnData,
    obsm_key: str = "spatial",
    batch_key: str = "batch",
    slide: str | None = None,
    with_images: bool = True,
    title_prefix: bool = True,
    axes: Sequence[plt.Axes] | None = None,
    **kwargs,
):
    """Spatial feature plot wrapping scanpy.pl.spatial for multiple batches/sections.

    Args:
        adata (anndata.AnnData): Annotated data matrix.
        obsm_key (str): Key in adata.obsm containing spatial coordinates.
        bacth_key (str): Key in adata.obs containing batch information.
        library_id (str): Library id.
        **kwargs: Additional arguments for scanpy.pl.spatial.

    Returns:
        matplotlib.figure.Figure: Figure.

    """
    assert obsm_key in adata.obsm.keys(), f"{obsm_key} not found in adata.obsm"
    assert batch_key in adata.obs.keys(), f"{batch_key} not found in adata.obs"
    batches = adata.obs[batch_key].cat.categories  # type:ignore
    if axes is not None:
        assert len(axes) == len(batches), "axes length must match the number of batches"
        kwargs["show"] = False
    else:
        axes = [None] * len(batches)  # type:ignore

    save = kwargs.pop("save", False)
    is_save_path = isinstance(save, str) or isinstance(save, Path)
    is_save_list = isinstance(save, list)
    if is_save_path:
        save = save.lstrip("_")
    else:
        kwargs["save"] = save

    if slide is None:
        feats_to_plot = kwargs.get("color")
        feats_to_plot = [feats_to_plot] if isinstance(feats_to_plot, str) else feats_to_plot
        title = kwargs.pop("title", [])
        if title and len(title) == len(batches) * len(feats_to_plot):  # type:ignore
            title_prefix = False
            title = np.asarray(title).reshape(len(batches), len(feats_to_plot))  # type:ignore
        else:
            logger.warning("Title not provided, or length does not match the number of batches. Using feature names as title.")
            title = feats_to_plot

        return_axes = []
        for i, batch in enumerate(batches):
            _adata = adata[adata.obs[batch_key] == batch]
            if is_save_path:
                kwargs["save"] = f"_{batch}_{save}"
            elif is_save_list:
                kwargs["save"] = save[i]

            if title_prefix:
                title_ = [f"{batch} " + _title for _title in feats_to_plot]  # type:ignore
            else:
                title_ = title[i]

            kwargs['ax'] = axes[i]
            if with_images:
                ax = sc.pl.spatial(_adata, library_id=batch, title=title_, **kwargs)  # type:ignore
            else:
                ax = sc.pl.embedding(_adata, basis=obsm_key, title=title_, **kwargs)
            return_axes.append(ax)
        return return_axes
    else:
        _adata = adata[adata.obs[batch_key] == slide]
        if with_images:
            return sc.pl.spatial(_adata, library_id=slide, **kwargs)
        return sc.pl.embedding(_adata, basis=obsm_key, **kwargs)


def plot_spatial_pie_charts(
    adata, spatial_key="spatial", library_id=None, deconv_key="deconv",
    size=1.,
):
    """
    Plot spatial pie charts of cell type proportions

    Args:
        adata (anndata.AnnData): Annotated data matrix.
        spatial_key (str): Key in adata.obsm containing spatial coordinates.
        library_id (str): Library id.
        deconv_key (str): Key in adata.obsm containing deconvolution results.
        size (float): Size of the pie chart.
    """
    spatial_coords = adata.obsm[spatial_key].copy()
    # flip y axis
    spatial_coords[:, 1] *= -1
    deconv_results = adata.obsm[deconv_key]
    if library_id is None:
        try:
            library_id = list(adata.uns["spatial"].keys())[0]
        except Exception:
            raise ("Library id not found")
    size *= (
        adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] / 2
    )

    fig, ax = plt.subplots()
    for i in range(len(adata)):
        x, y = spatial_coords[i]
        deconv_data = deconv_results.iloc[i]

        ax.pie(deconv_data, radius=size,
               center=(x, y), frame=True)

    ax.set_aspect("equal", "box")
    ax.set_title("Spatial Pie Chart of Cell Type Proportions")
    # remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # remove frame
    ax.set_frame_on(False)
    # remove grid
    ax.grid(False)
    plt.show()


def plot_posterior_mu_vs_data(mu, data, log10=True):
    """
    (from cell2location.utils_plot.plot_posterior_mu_vs_data)
    Plot the posterior expected value against the data.

    Args:
        mu (np.ndarray): Posterior expected value.
        data (np.ndarray): Data.
        log10 (bool): Whether to plot in log10 scale.
    """

    if not isinstance(data, np.ndarray):
        data = data.toarray()

    if log10:
        plt.hist2d(
            np.log10(data.flatten() + 1),
            np.log10(mu.flatten() + 1),
            bins=50,
            norm=mpl.colors.LogNorm(),
        )
    else:
        plt.hist2d(
            data.flatten(),
            mu.flatten(),
            bins=50,
            norm=mpl.colors.LogNorm(),
        )
    plt.gca().set_aspect("equal", adjustable="box")
    ind = "" if not log10 else ", log10"
    plt.xlabel(f"Data{ind}")
    plt.ylabel(f"Posterior expected value{ind}")
    plt.title("Reconstruction accuracy")
    plt.tight_layout()


def plot_domain_summary(adata, domain_key, cell_type_names, figsize, show=True):
    """
    Plot domain summary.

    Args:
        adata (anndata.AnnData): Annotated data matrix.
        domain_key (str): Key in adata.obs containing domain information.
        cell_type_names (list): List of cell type names.
        figsize (tuple): Figure size.
        show (bool): Whether to show the plot.

    Returns:
        matplotlib.figure.Figure: Figure.
    """
    df = adata.obs[cell_type_names + [domain_key]]
    df = df.groupby("domain").agg("mean")  # type:ignore
    df: pd.DataFrame
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    sc.pl.spatial(adata, color=domain_key, ax=axes[0], show=False, size=1.3)
    df = df.sort_values("domain", ascending=False)  # type:ignore
    df.plot(kind="barh", stacked=True, figsize=figsize, width=0.8, ax=axes[1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)
    if show:
        plt.show()
    return fig


def plot_domain_summary_single_ct(
    adata,
    domain_key,
    cell_type_name,
    figsize,
    show=True,
    plot_func=sc.pl.spatial,
    **kwargs,
):
    df = adata.obs[[cell_type_name, domain_key]]
    df: pd.DataFrame
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_func(adata, color=domain_key, ax=axes[0], show=False, **kwargs)
    df = df.sort_values("domain", ascending=False)  # type:ignore
    sns.barplot(data=df, x=domain_key, y=cell_type_name, ax=axes[1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)
    if show:
        plt.show()
    return fig


def plot_single_domain_summary(
    adata, domain, domain_key, cell_type_names, figsize, show=True
):
    df = adata.obs[cell_type_names + [domain_key]]
    df = df[df[domain_key] == domain]
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    df.plot(kind="barh", stacked=True, figsize=figsize, width=0.8, ax=axes)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)
    if show:
        plt.show()
    return fig


def set_cmap(adata, obs_key, color_map, reverse=False):
    cmap = plt.get_cmap(color_map)

    # Get unique categories
    unique_categories = adata.obs[obs_key].cat.categories
    n_categories = len(unique_categories)
    cat_codes = np.linspace(0, 0.9, n_categories)
    if reverse:
        cat_codes = cat_codes[::-1]

    if isinstance(cmap, mpl.colors.ListedColormap):  # discrete colormap
        if n_categories > cmap.N:
            raise ValueError(
                "Can't fit data categories into given discrete colormap, too many categories."
            )
        colors = cmap(cat_codes)  # get all available colors
        colors = colors[:n_categories]  # subset for needed categories
    else:  # continuous colormap
        colors = cmap(cat_codes)  # create needed colors from colormap

    hex_colors = [colors[i][:3] for i in range(colors.shape[0])]
    hex_colors = [mpl.colors.rgb2hex(c) for c in hex_colors]
    adata.uns[f"{obs_key}_colors"] = hex_colors


def plot_trajectory(adata, pseudotime, library_id=None, location='spatial', clusterlabels='domain', gridnum=10, pointsize=5,
                    arrowlength=0.2, arrow_color='red', save_endpoints=False,
                    **kwargs):
    from matplotlib.patches import FancyArrow

    # Prepare dataframe from the provided `location` and `pseudotime`
    info = adata.obs[[pseudotime, clusterlabels]]
    info.rename(columns={pseudotime: "pseudotime"}, inplace=True)
    if isinstance(location, str):
        info["sdimx"] = 1 * adata.obsm[location][:, 0]
        info["sdimy"] = 1 * adata.obsm[location][:, 1]
        # adata.obsm['spatial_filp'] = adata.obsm['spatial'].copy()
        # adata.obsm['spatial_filp'][:, 1] = -adata.obsm['spatial_filp'][:, 1]
    else:
        info[["sdimx", "sdimy"]] = adata.obs[location]
        info["sdimx"] = -1 * info["sdimx"]

    min_x = info['sdimx'].min()
    min_y = info['sdimy'].min()
    max_x = info['sdimx'].max()
    max_y = info['sdimy'].max()

    x_anchor = np.linspace(min_x, max_x, gridnum + 1)
    y_anchor = np.linspace(min_y, max_y, gridnum + 1)

    direction_pseudotime_point = {}
    endpoints = {}
    count = 0
    # Perform the similar grid based assignment and arrow plotting
    for num_x in range(gridnum):
        for num_y in range(gridnum):
            filter_x = info[(info['sdimx'] >= x_anchor[num_x]) & (info['sdimx'] <= x_anchor[num_x + 1])]
            filter_y = info[(info['sdimy'] >= y_anchor[num_y]) & (info['sdimy'] <= y_anchor[num_y + 1])]
            points_in_grid = pd.merge(filter_x, filter_y)

            if len(points_in_grid) > 1:
                count += 1
                min_point = points_in_grid[points_in_grid['pseudotime'] == points_in_grid['pseudotime'].min()]
                max_point = points_in_grid[points_in_grid['pseudotime'] == points_in_grid['pseudotime'].max()]
                direction = {'start': min_point[['sdimx', 'sdimy']].values[0],
                             'end': max_point[['sdimx', 'sdimy']].values[0]}
                if save_endpoints:
                    # save the start cluster label and end cluster label
                    endpoints[count] = {'start': min_point[clusterlabels].values[0],
                                        'end': max_point[clusterlabels].values[0]}
                direction_pseudotime_point[count] = direction

    if endpoints:
        adata.uns['endpoints'] = pd.DataFrame(endpoints)
    # Begin plotting
    save = kwargs.pop('save', False)
    if location == 'X_umap':
        g = sc.pl.umap(adata, color=clusterlabels, show=False, **kwargs)
    else:
        g = sc.pl.spatial(adata,
                          color=clusterlabels,
                          show=False,
                          img_key=None,
                          library_id=library_id, **kwargs)[0]
    ax = g.axes

    arrowlength = arrowlength if location != 'X_umap' else 0.001
    for _, value in direction_pseudotime_point.items():
        arrow = FancyArrow(*value['start'], *value['end'] - value['start'],
                           width=arrowlength,
                           head_width=75 * arrowlength,
                           head_length=75 * arrowlength,
                           color=arrow_color, alpha=1.)
        ax.add_patch(arrow)

    if save:
        file = 'trajectory.png' if save is True else save
        plt.savefig(sc.settings.figdir / file, bbox_inches='tight')
    plt.show()


def plot_trajectory_curve(
    adata, pseudotime, library_id=None, location="spatial", clusterlabels="domain",
    gridnum=10,
    arrow_color="red",
    save_endpoints=False,
    arrowlength=0.2,
    n_quantiles=4,
    order=3,
    num_t=10,
    **kwargs
):
    from scipy.interpolate import make_interp_spline

    # use plot_trajectory to plot the curve trajectory instead of arrows
    # Prepare dataframe from the provided `location` and `pseudotime`
    info = adata.obs[[pseudotime, clusterlabels]]
    info.rename(columns={pseudotime: "pseudotime"}, inplace=True)
    if isinstance(location, str):
        info["sdimx"] = 1 * adata.obsm[location][:, 0]
        info["sdimy"] = 1 * adata.obsm[location][:, 1]
    else:
        info[["sdimx", "sdimy"]] = adata.obs[location]
        info["sdimx"] = -1 * info["sdimx"]

    min_x = info["sdimx"].min()
    min_y = info["sdimy"].min()
    max_x = info["sdimx"].max()
    max_y = info["sdimy"].max()

    x_anchor = np.linspace(min_x, max_x, gridnum + 1)
    y_anchor = np.linspace(min_y, max_y, gridnum + 1)

    direction_pseudotime_point = {}
    endpoints = {}
    count = 0
    # Perform the similar grid based assignment and arrow plotting
    for num_x in range(gridnum):
        for num_y in range(gridnum):
            filter_x = info[(info["sdimx"] >= x_anchor[num_x]) & (info["sdimx"] <= x_anchor[num_x + 1])]
            filter_y = info[(info["sdimy"] >= y_anchor[num_y]) & (info["sdimy"] <= y_anchor[num_y + 1])]
            points_in_grid = pd.merge(filter_x, filter_y)

            if len(points_in_grid) > 1:
                count += 1
                # save quantile points by sorting the pseudotime
                sorted_points = points_in_grid.sort_values('pseudotime')
                quantiles = (len(sorted_points) * np.linspace(0, 1, n_quantiles + 1, endpoint=False)).astype(int)
                quantiles = np.unique(quantiles)
                quantile_points = sorted_points.iloc[quantiles]

                direction = quantile_points[['sdimx', 'sdimy']].values
                direction_pseudotime_point[count] = direction

                if save_endpoints:
                    # save the start cluster label and end cluster label
                    endpoints[count] = {f'node_{i}': quantile_points[clusterlabels].values[i] for i in range(n_quantiles)}
    if endpoints:
        adata.uns['endpoints'] = pd.DataFrame(endpoints)
    # Begin plotting
    save = kwargs.pop("save", False)
    if location == "X_umap":
        g = sc.pl.umap(adata, color=clusterlabels, show=False, **kwargs)
    else:
        g = sc.pl.spatial(adata, color=clusterlabels, show=False, img_key=None, library_id=library_id, **kwargs)[0]
    ax = g.axes

    # fit a smooth curve through the quantile points
    for _, value in direction_pseudotime_point.items():
        # ensure the number of quantile points is enough to fit a curve
        if len(value) <= order:
            continue
        # fit a smooth curve through the quantile points using parametric spline
        x = value[:, 0]
        y = value[:, 1]
        t = np.linspace(0, 1, len(x))
        tnew = np.linspace(0, 1, num_t)
        splx = make_interp_spline(t, x, k=min(order, len(x) - 1))
        sply = make_interp_spline(t, y, k=min(order, len(y) - 1))
        xnew = splx(tnew)
        ynew = sply(tnew)
        ax.plot(xnew, ynew, color=arrow_color)
        # add arrow head
        arrow = plt.arrow(xnew[-2], ynew[-2], xnew[-1] - xnew[-2], ynew[-1] - ynew[-2],
                          width=arrowlength,
                          head_width=75 * arrowlength,
                          head_length=75 * arrowlength,
                          fc=arrow_color,
                          ec=arrow_color)
        ax.add_patch(arrow)

    if save:
        file = "trajectory.png" if save is True else save
        plt.savefig(sc.settings.figdir / file, bbox_inches="tight")
    plt.show()
    return ax
