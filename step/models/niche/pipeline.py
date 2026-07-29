from collections import Counter

import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from typing import Literal, List
from scipy.spatial import KDTree
from sklearn.cluster import SpectralClustering
from tqdm import tqdm

from .helper import compute_kernel_block_wise, nystrom_spectral_clustering
from .helper import min_max_scale


class MicroArc:
    def __init__(
        self,
        metadata: pd.DataFrame,
        label_col: str,
        batch_col: str,
        spatial_cols: tuple = ("x", "y"),
        coherence_col: str = "cosine",
        coherence_threshold: float = 0.4,
        radius: float = 100,
        edge_threshold: float = None,
        n_clusters: int = 5,
        n_iter: int = 5,
    ):
        """
        Initialize MicroArchitecture with metadata and parameters
        """
        self.metadata = metadata
        self.label_col = label_col
        self.batch_col = batch_col
        self._spatial_cols = spatial_cols
        self.coherence_col = coherence_col
        self.coherence_threshold = coherence_threshold
        self.radius = radius
        self.edge_threshold = (
            edge_threshold if edge_threshold is not None else radius / 2
        )
        self.n_clusters = n_clusters
        self.n_iter = n_iter

        # Initialize containers
        self.graphs = []
        self.center_indices = []
        self.neighbor_indices = []
        self.clusters = None
        self.kernel_matrix = None

        all_cell_types = sorted(metadata[label_col].unique())
        temp_graph = nx.Graph(
            [(i, (i + 1) % len(all_cell_types)) for i in range(len(all_cell_types))]
        )
        self.global_positions = dict(
            zip(all_cell_types, nx.circular_layout(temp_graph).values())
        )

    @property
    def spatial_cols(self):
        return list(self._spatial_cols)

    @classmethod
    def restore_from_metadata(
        cls, metadata: pd.DataFrame, center_pattern_key: str, **kwargs
    ):
        """
        Restore MicroArchitecture object from metadata
        """
        assert center_pattern_key in metadata.columns, (
            f"Pattern key {center_pattern_key} not found in metadata"
        )
        niche = cls(metadata, **kwargs)
        center_indices = metadata.loc[~metadata[center_pattern_key].isna()].index
        niche.coherence_threshold = metadata[metadata[center_pattern_key].notna()][
            niche.coherence_col
        ].max()
        niche.create_graphs(center_indices)
        niche.clusters = metadata.loc[niche.center_indices, center_pattern_key].values
        return niche

    def create_graphs(self, query: pd.Index):
        """
        Create niche graphs from query indices
        """
        from scipy.spatial.distance import pdist, squareform

        # Filter centers
        low_coherence_centers = self.metadata.loc[query][
            self.metadata.loc[query][self.coherence_col] < self.coherence_threshold
        ]
        print(f"Found {len(low_coherence_centers)} low coherence centers")

        # Process each batch separately
        for batch in self.metadata[self.batch_col].unique():
            # Filter data for current batch
            batch_data = self.metadata[self.metadata[self.batch_col] == batch]
            batch_centers = low_coherence_centers[
                low_coherence_centers[self.batch_col] == batch
            ]

            if len(batch_centers) == 0:
                continue

            # Find neighbors within batch
            tree = KDTree(batch_data[self.spatial_cols].values)
            center_coords = batch_centers[self.spatial_cols].values
            all_neighbors_indices = tree.query_ball_point(center_coords, self.radius)

            # Create niche graphs for current batch
            for center_idx, neighbor_indices in tqdm(
                zip(batch_centers.index, all_neighbors_indices),
                total=len(batch_centers),
                desc=f"Creating graphs for batch {batch}",
            ):
                if len(neighbor_indices) < 3:
                    continue

                neighbors = batch_data.iloc[neighbor_indices]
                points = neighbors[self.spatial_cols].values

                # Calculate all pairwise distances once
                distances = squareform(pdist(points))

                # Create graph
                G = nx.Graph()
                G.add_nodes_from(neighbors.index)
                nx.set_node_attributes(
                    G, neighbors[self.label_col].to_dict(), "cell_type"
                )

                # Get edges from distance matrix
                rows, cols = np.where(
                    (distances <= self.edge_threshold) & (distances > 0)
                )
                mask = rows < cols
                edges = zip(neighbors.index[rows[mask]], neighbors.index[cols[mask]])
                G.add_edges_from(edges)

                if len(G.edges) > 4:
                    self.graphs.append(G)
                    self.center_indices.append(center_idx)
                    self.neighbor_indices.append(neighbors.index)

        if not self.graphs:
            raise ValueError("No valid niche graphs could be created")

        return self

    def compute_kernel_matrix(self, save: bool = False):
        kernel_matrix = compute_kernel_block_wise(self.graphs)
        if save:
            self.kernel_matrix = kernel_matrix
        return kernel_matrix

    def compute_kernel_and_cluster(
        self,
        method: Literal["full", "approx"] = "full",
        save_kernel_matrix: bool = False,
        **kwargs,
    ):
        """
        Compute graph kernel and perform clustering
        """
        print("Computing graph kernels...")
        kernel_matrix = self.compute_kernel_matrix(save=save_kernel_matrix)

        print("Performing clustering...")
        if method == "full":
            clustering = SpectralClustering(
                n_clusters=min(self.n_clusters, len(self.graphs)),
                affinity="precomputed",
                random_state=42,
            )
            self.clusters = clustering.fit_predict(kernel_matrix)
        elif method == "approx":
            self.clusters = nystrom_spectral_clustering(
                kernel_matrix, min(self.n_clusters, len(self.graphs)), **kwargs
            )

        return self

    def analyze_patterns(self, enrichment_threshold=2.0, specificity_threshold=0.8):
        """
        Analyze patterns in clusters
        """
        if self.clusters is None:
            raise ValueError("Run compute_kernel_and_cluster first")

        # Get unique slices from the data
        centers_data = self.metadata.loc[self.center_indices]
        slices = sorted(centers_data[self.batch_col].unique())

        # Get total counts per slice
        total_per_slice = {}
        for slice_id in slices:
            total_per_slice[slice_id] = sum(
                1
                for i in range(len(self.center_indices))
                if centers_data.iloc[i][self.batch_col] == slice_id
            )

        # Analyze cluster composition
        cluster_stats = []
        for k in range(self.n_clusters):
            cluster_members = np.where(self.clusters == k)[0]
            total_count = len(cluster_members)

            # Count and calculate proportions for each slice
            slice_counts = {}
            slice_proportions = {}

            for slice_id in slices:
                count = sum(
                    1
                    for i in cluster_members
                    if centers_data.iloc[i][self.batch_col] == slice_id
                )
                slice_counts[f"{slice_id}_count"] = count
                # Calculate proportion within slice
                slice_proportions[f"{slice_id}_proportion"] = (
                    count / total_per_slice[slice_id]
                    if total_per_slice[slice_id] > 0
                    else 0
                )

            # Determine pattern type
            pattern_type = self._determine_pattern_type(
                slice_proportions, slices, enrichment_threshold, specificity_threshold
            )

            # Compile statistics
            stats = {
                "cluster": k,
                "total_niches": total_count,
                **slice_counts,
                **slice_proportions,
                "pattern_type": pattern_type,
            }
            cluster_stats.append(stats)

        return pd.DataFrame(cluster_stats)

    def _determine_pattern_type(
        self, proportions, slices, enrichment_threshold, specificity_threshold
    ):
        """Helper method to determine pattern type"""
        # Check for slice-specific patterns
        for slice_id in slices:
            if proportions[f"{slice_id}_proportion"] >= specificity_threshold:
                return f"{slice_id}-specific"

        # Check for enrichment
        for slice1 in slices:
            prop1 = proportions[f"{slice1}_proportion"]
            is_enriched = True

            for slice2 in slices:
                if slice1 != slice2:
                    prop2 = proportions[f"{slice2}_proportion"]
                    if prop1 < enrichment_threshold * prop2:
                        is_enriched = False
                        break

            if is_enriched:
                return f"{slice1}-enriched"

        return "Shared"

    def get_graph(self, idx):
        """Get graph and associated data by index"""
        return {
            "graph": self.graphs[idx],
            "center": self.metadata.loc[self.center_indices[idx]],
            "neighbors": self.metadata.loc[self.neighbor_indices[idx]],
        }

    def get_average_pattern(
        self,
        pattern_idx: int,
        batch_key: str = None,
        use_frequency: bool = True,
        min_edge_count: int = 1,
        ax: plt.Axes = None,
        node_size_range: tuple = (50, 500),
        edge_width_range: tuple = (2, 10),
        k: float = None,
        edge_curve: float = 0.2,
        return_summary: bool = False,
        cmap: List[str] | None = None,
        return_fig: bool = False,
    ):
        """
        Analyze average cell type composition and connections in a pattern

        Parameters:
        -----------
        pattern_idx : int
            Pattern/cluster index to analyze
        batch_key : str
            If provided, only analyze cells from this batch
        use_frequency : bool
            Whether to use frequencies (True) or counts (False)
        min_edge_count : int
            Minimum number of edges to show in network plot
        ax : plt.Axes
            Axes to plot on
        node_size_range : tuple
            (min, max) sizes for nodes
        edge_width_range : tuple
            (min, max) widths for edges
        k : float
            Spring layout parameter, larger k means more spread out
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        pattern_members = np.where(self.clusters == pattern_idx)[0]
        pattern_ind = [self.center_indices[idx] for idx in pattern_members]
        print(f"Found {len(pattern_members)} cells with pattern {pattern_idx}")
        if batch_key is not None:
            pattern_members = pattern_members[
                self.metadata.loc[pattern_ind, self.batch_col] == batch_key
            ]

            if not len(pattern_members) > 0:
                ax.text(0.5, 0.5, "No examples in this batch", ha="center", va="center")
                ax.axis("off")
                return None

        # 1. Summarize cell type counts

        neighbor_ind = np.concatenate(
            [self.neighbor_indices[idx] for idx in pattern_members]
        )
        neighbors = self.metadata.loc[neighbor_ind]
        counts = neighbors[self.label_col].value_counts(normalize=use_frequency)
        counts = min_max_scale(counts, *node_size_range)
        node_sizes = {
            ct: counts.get(ct, node_size_range[0])
            for ct in self.metadata[self.label_col].unique()
        }

        # 2. Count cell type pair edges
        edges = np.concatenate([self.graphs[idx].edges() for idx in pattern_members])

        cell_pairs = [self.metadata.loc[edge, self.label_col].values for edge in edges]

        edge_pairs = [tuple(sorted(pair)) for pair in cell_pairs]
        edge_counts = Counter(edge_pairs)

        G_avg = nx.Graph()

        for cell_type in node_sizes:
            G_avg.add_node(cell_type)

        for (cell1, cell2), count in edge_counts.items():
            if count >= min_edge_count:
                G_avg.add_edge(cell1, cell2, weight=count)

        pos = {node: self.global_positions[node] for node in G_avg.nodes()}
        if cmap is None:
            import seaborn as sns

            cmap = sns.palettes.color_palette("tab20")[: len(self.global_positions)]

        nx.draw_networkx_nodes(
            G_avg,
            pos,
            node_size=[node_sizes[node] for node in G_avg.nodes()],
            node_color=cmap,
            alpha=0.6,
            ax=ax,
        )

        # Draw edges with curved paths
        if G_avg.edges():
            edge_weights = np.array([G_avg[u][v]["weight"] for u, v in G_avg.edges()])
            edge_widths = min_max_scale(edge_weights, *edge_width_range)

            # Draw each edge with a curved path
            for (node1, node2), width in zip(G_avg.edges(), edge_widths):
                # Get node positions
                pos1 = pos[node1]
                pos2 = pos[node2]

                # Create curved connection
                connection = patches.ConnectionStyle.Arc3(rad=edge_curve)

                # Draw edge
                edge_patch = patches.FancyArrowPatch(
                    pos1,
                    pos2,
                    connectionstyle=connection,
                    arrowstyle="-",
                    linewidth=width,
                    alpha=0.4,
                    color="gray",
                )
                ax.add_patch(edge_patch)

        label_pos = {node: (coord[0], coord[1] + 0.05) for node, coord in pos.items()}

        nx.draw_networkx_labels(
            G_avg,
            label_pos,
            ax=ax,
            font_size=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
        )

        ax.margins(x=0.2, y=0.2)
        # Add legend for node sizes

        title = f"Average Pattern {pattern_idx} Structure\n"
        if batch_key is not None:
            title += f"({batch_key}, "
        title += (
            f"{len(pattern_members)} niches, {len(edge_counts)} unique connections)"
        )
        ax.set_title(title)
        ax.axis("off")

        if return_summary:
            summary = {
                "node_sizes": node_sizes,
                "edge_counts": edge_counts,
                "total_niches": len(pattern_members),
                "total_connections": sum(edge_counts.values()),
            }

            return summary

    def plot_patterns(
        self,
        cluster_stats: pd.DataFrame | None = None,
        show_edges=False,
        figsize_multiplier=(24, 4),
        min_edge_count: int = 1,
        use_frequency: bool = True,
        **kwargs,
    ):
        """
        Visualize patterns with emphasis on sharing between slices.
        """
        # Get unique slices from center indices
        slices = self.metadata.loc[self.center_indices, self.batch_col].unique()
        n_slices = len(slices)
        n_cols = 3 + n_slices

        # Create global color mapping for cell types
        unique_cell_types = sorted(self.metadata[self.label_col].unique())
        n_cell_types = len(unique_cell_types)
        cmap = plt.cm.tab20(np.linspace(0, 1, n_cell_types))
        cell_type_to_color = cmap

        # Create figure
        fig = plt.figure(
            figsize=(figsize_multiplier[0], figsize_multiplier[1] * self.n_clusters)
        )

        for k in range(self.n_clusters):
            cluster_members = np.where(self.clusters == k)[0]

            if cluster_stats is not None:
                stats = cluster_stats.iloc[k]

                ax_count = fig.add_subplot(self.n_clusters, n_cols, k * n_cols + 1)

                # Get counts for each slice
                slice_counts = [stats[f"{s}_count"] for s in slices]

                bars = ax_count.bar(slices, slice_counts)
                ax_count.set_xticks(range(len(slices)))
                ax_count.set_xticklabels(slices, rotation=45, ha="right")
                ax_count.set_title(
                    f"Pattern {k}: {stats['pattern_type']}\nTotal n={stats['total_niches']}"
                )

                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax_count.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                    )

                ax_prop = fig.add_subplot(self.n_clusters, n_cols, k * n_cols + 2)

                # Get proportions for each slice
                slice_props = [stats[f"{s}_proportion"] for s in slices]

                bars = ax_prop.bar(slices, slice_props)
                ax_prop.set_xticks(range(len(slices)))
                ax_prop.set_xticklabels(slices, rotation=45, ha="right")
                ax_prop.set_title("Proportions within slices")

                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax_prop.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                    )

            ax_network = fig.add_subplot(self.n_clusters, n_cols, k * n_cols + 3)

            self.get_average_pattern(
                k,
                use_frequency=use_frequency,
                min_edge_count=min_edge_count,
                cmap=cell_type_to_color,
                ax=ax_network,
                **kwargs,
            )
            ax_network.set_title("Average Pattern Structure")

            for i, slice_id in enumerate(slices):
                members = [
                    idx
                    for idx in cluster_members
                    if self.metadata.loc[self.center_indices[idx], self.batch_col]
                    == slice_id
                ]

                ax = fig.add_subplot(self.n_clusters, n_cols, k * n_cols + i + 4)
                if members:
                    self.get_average_pattern(
                        k,
                        use_frequency=use_frequency,
                        min_edge_count=min_edge_count,
                        batch_key=slice_id,
                        ax=ax,
                        cmap=cell_type_to_color,
                        **kwargs,
                    )

                else:
                    ax.text(0.5, 0.5, "No examples", ha="center", va="center")

                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()

    def plot_spatial_pattern_distribution(
        self,
        metadata,
        background_cells,
        background_label="Background",
        batch_key=None,
    ):
        """
        Plot spatial distribution of self.patterns in original tumor space.
        """

        if batch_key is None:
            batch_key = self.batch_col

        self.graphs = self.graphs
        clusters = self.clusters
        centers = metadata.loc[self.center_indices]
        n_batches = len(metadata[batch_key].unique())

        # Create figure with two subplots (one for each tumor)
        fig, axes = plt.subplots(1, n_batches, figsize=(6 * n_batches, 4.5))

        # Color map for patterns
        n_patterns = len(np.unique(clusters))
        pattern_colors = sns.color_palette("husl", n_patterns)

        for idx, sample in enumerate(metadata[batch_key].unique()):
            ax = axes[idx]

            # Plot all cells as background
            sample_cells = metadata[metadata[batch_key] == sample]
            sample_bg_mask = sample_cells.index.isin(background_cells)

            bg_meta_bottom = sample_cells.loc[
                ~sample_bg_mask & ~sample_cells.index.isin(centers.index)
            ]
            ax.scatter(
                bg_meta_bottom["x"],
                bg_meta_bottom["y"],
                rasterized=True,
                c="lightgray",
                s=0.5,
                alpha=0.1,
                label="All cells",
            )

            bg_meta = sample_cells.loc[sample_bg_mask]
            ax.scatter(
                bg_meta["x"],
                bg_meta["y"],
                rasterized=True,
                c="lightgray",
                s=1,
                alpha=0.5,
                label=background_label,
            )

            # Plot pattern centers with different colors
            for pattern in range(n_patterns):
                # Find centers for this pattern and sample
                pattern_centers = centers.iloc[clusters == pattern]
                pattern_centers = pattern_centers.loc[
                    pattern_centers.index.isin(sample_cells.index)
                ]

                if not pattern_centers.empty:
                    x = pattern_centers["x"]
                    y = pattern_centers["y"]
                    ax.scatter(
                        x,
                        y,
                        c=[pattern_colors[pattern]],
                        s=4,
                        label=f"Pattern {pattern}",
                        rasterized=True,
                        alpha=0.7,
                    )

                    # Optionally: draw circles to show self.radius
                    for cx, cy in zip(x, y):
                        circle = plt.Circle(
                            (cx, cy),
                            radius=100,
                            color=pattern_colors[pattern],
                            fill=False,
                            alpha=0.2,
                        )
                        ax.add_patch(circle)

            ax.set_title(f"{sample} Micro-Architecture Distribution")
            if idx == n_batches - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.set_aspect("equal")
            ax.axis("off")

        fig.tight_layout()
        return fig
