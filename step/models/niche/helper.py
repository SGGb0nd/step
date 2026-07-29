import numpy as np
from tqdm import tqdm
from grakel import WeisfeilerLehman, VertexHistogram


def compute_kernel_block_wise(graphs, block_size=5000):
    n_graphs = len(graphs)
    n_blocks = (n_graphs + block_size - 1) // block_size

    # Initialize with float16
    kernel_matrix = np.zeros((n_graphs, n_graphs))

    wl = WeisfeilerLehman(
        n_iter=3, base_graph_kernel=VertexHistogram, normalize=True)

    graphs_for_kernel = [
        (list(G.edges()),
         {node: G.nodes[node]['cell_type'] for node in G.nodes()})
        for G in tqdm(graphs, desc="Preparing graphs")
    ]

    total_blocks = sum(n_blocks - i for i in range(n_blocks))

    with tqdm(total=total_blocks, desc="Computing kernel blocks") as pbar:
        for i in range(n_blocks):
            start_i = i * block_size
            end_i = min((i + 1) * block_size, n_graphs)

            for j in range(i, n_blocks):
                start_j = j * block_size
                end_j = min((j + 1) * block_size, n_graphs)

                if i == j:
                    # When computing the diagonal block, compute the kernel on the block only.
                    block_kernel = wl.fit_transform(
                        graphs_for_kernel[start_i:end_i]
                    )
                    # Here, block_kernel is square where diagonal elements reflect self-similarity.
                    kernel_matrix[start_i:end_i, start_i:end_i] = block_kernel
                else:
                    # For off-diagonal blocks, concatenate both blocks.
                    block_kernel = wl.fit_transform(
                        graphs_for_kernel[start_i:end_i] +
                        graphs_for_kernel[start_j:end_j]
                    )
                    n_i = end_i - start_i
                    # The cross kernel is in the upper-right and lower-left.
                    block = block_kernel[:n_i, n_i:].astype(np.float16)

                    kernel_matrix[start_i:end_i, start_j:end_j] = block
                    kernel_matrix[start_j:end_j, start_i:end_i] = block.T

                pbar.update(1)
                pbar.set_postfix({
                    'block': f'({i},{j})',
                    'size': f'{end_i-start_i}x{end_j-start_j}'
                })

    return kernel_matrix


def nystrom_spectral_clustering(kernel_matrix, n_clusters, n_samples=5000):
    """
    Nyström approximation for spectral clustering
    """
    n = kernel_matrix.shape[0]

    idx = np.random.choice(n, min(n_samples, n), replace=False)

    K_nm = kernel_matrix[:, idx]
    K_mm = kernel_matrix[idx][:, idx]

    eigenvalues, eigenvectors = np.linalg.eigh(K_mm.astype(np.float32))

    idx_sorted = np.argsort(eigenvalues)[::-1][:n_clusters]
    eigenvalues = eigenvalues[idx_sorted]
    eigenvectors = eigenvectors[:, idx_sorted]

    U = np.sqrt(n) * K_nm @ eigenvectors @ np.diag(1.0 / eigenvalues)
    U = U / np.linalg.norm(U, axis=1, keepdims=True)

    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(U)

    return clusters


def min_max_scale(X, min_val, max_val):
    if isinstance(X, int):
        return X

    normed = (X - X.min()) / (X.max() - X.min())
    return normed * (max_val - min_val) + min_val
