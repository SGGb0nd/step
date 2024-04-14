from typing import Any

import numpy as np


def mclust_R(data, num_cluster, modelNames="EEE", random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects

    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri

    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r["set.seed"]
    r_random_seed(random_seed)
    rmclust = robjects.r["Mclust"]

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(
        data), num_cluster, modelNames)
    mclust_res = np.array(res[-2])
    return mclust_res


def kmeans(n_clusters, data, seed=None):
    from sklearn.cluster import KMeans

    return KMeans(n_clusters, random_state=seed,).fit_predict(data)


class ClusterGeneric:
    """
    A generic clustering class that supports different clustering methods.

    Attributes:
        method (str): The clustering method to use.
        seed (int): The random seed to use.

    """

    def __init__(self, method="kmeans", seed=None) -> None:
        self.method = method
        self.seed = seed

    def __call__(self, n_clusters, data, *args: Any, **kwds: Any) -> Any:
        """
        Perform clustering based on the specified method.

        Args:
            n_clusters (int): The number of clusters to create.
            data (array-like): The input data to be clustered.
            *args: Additional positional arguments.
            **kwds: Additional keyword arguments.

        Returns:
            Any: The result of the clustering operation.

        """
        if self.method == "kmeans":
            return kmeans(n_clusters, data, seed=self.seed)
        elif self.method == "mclust":
            return mclust_R(num_cluster=n_clusters, data=data)
