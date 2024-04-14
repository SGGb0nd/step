import numpy as np


def nnls_deconv(
    signatures,
    observations,
    n_jobs: int = 1,
):
    """Perform non-negative least squares (NNLS) deconvolution.

    Args:
        signatures (np.ndarray): The signature matrix.
        observations (np.ndarray): The observation matrix.
        n_jobs (int, optional): Number of jobs. Defaults to 1.

    Returns:
        np.ndarray: The deconvoluted matrix.
    """
    from sklearn.linear_model import LinearRegression
    nnls = LinearRegression(n_jobs=n_jobs, fit_intercept=False, positive=True)

    return np.array(
        [
            nnls.fit(signatures.T, obs).coef_
            for obs in observations
        ]
    )
