import numpy as np
from jaxtyping import Real, jaxtyped
from beartype import beartype
from numba import njit

from .utils import get_cluster_info

@njit
def _sample_cadi_triplet(
    offsets,
    valid_clusterB_idxs,
    flat,
    cluster_sizes,
    rng: np.random.Generator,
):
    n_clusters = offsets.shape[0]
    n_validB = valid_clusterB_idxs.shape[0]

    # Sample cluster A
    cA = rng.integers(0, n_clusters)

    # Sample cluster B (must differ)
    cB = valid_clusterB_idxs[rng.integers(0, n_validB)]
    while cA == cB:
        cB = valid_clusterB_idxs[rng.integers(0, n_validB)]

    # Sample members
    sizeA = cluster_sizes[cA]
    sizeB = cluster_sizes[cB]

    offA = offsets[cA]
    offB = offsets[cB]

    x_idx = flat[offA + rng.integers(0, sizeA)]

    y_off = rng.integers(0, sizeB)
    z_off = rng.integers(0, sizeB)
    while y_off == z_off:
        z_off = rng.integers(0, sizeB)

    y_idx = flat[offB + y_off]
    z_idx = flat[offB + z_off]

    return x_idx, y_idx, z_idx

@njit
def _get_cosine(X, x_idx, y_idx, z_idx):
    d = X.shape[1]

    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0

    for j in range(d):
        v1 = X[y_idx, j] - X[x_idx, j]
        v2 = X[z_idx, j] - X[x_idx, j]

        dot += v1 * v2
        norm1 += v1 * v1
        norm2 += v2 * v2

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    res = dot / np.sqrt(norm1 * norm2)
    return res

@njit
def _cadi_kernel(
    X,
    Y,
    offsets,
    valid_clusterB_idxs,
    flat,
    cluster_sizes,
    n_triplets: int,
    rng: np.random.Generator,
):
    sum_sqr = 0.0

    for _ in range(n_triplets):

        # sample triplet
        x_idx, y_idx, z_idx = _sample_cadi_triplet(
            offsets,
            valid_clusterB_idxs,
            flat,
            cluster_sizes,
            rng
        )

        # compute cosines
        cosX = _get_cosine(X, x_idx, y_idx, z_idx)
        cosY = _get_cosine(Y, x_idx, y_idx, z_idx)

        diff = cosX - cosY
        sum_sqr += diff * diff

    return sum_sqr / (4.0 * n_triplets)

@jaxtyped(typechecker=beartype)
def CADI(
    X: Real[np.ndarray, "N d"],
    Y: Real[np.ndarray, "N d2"],
    labels: Real[np.ndarray, "N"],
    n_triplets: int = 0,
    random_seed: int | None | np.random.Generator = None,
) -> float:
    """
    Computes the Class Angular Distortion Index between a dataset X and a projection Y

    Parameters
    ----------
    X : ndarray of shape (N, d)
        (High-dimensional) dataset.
    Y : ndarray of shape (N, d2)
        (Low-dimensional) projection.
    labels : ndarray of shape (N,)
        Maps each vector to a class label.
    n_triplets : int, optional
        Number of inter-cluster triplets to sample. Defaults to N.
    random_seed : np.random.Generator, optional
        Random number generator for reproducibility.
    
    Returns
    -------
    loss : float
        Mean squared difference of cosines for inter-cluster triplets, normalized to [0,1].
    """
    if not isinstance(random_seed, np.random.Generator):
        rng = np.random.default_rng(random_seed)
    else:
        rng = random_seed

    if n_triplets == 0:
        n_triplets = X.shape[0] * 10

    cluster_info = get_cluster_info(labels)

    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    return _cadi_kernel(
        X,
        Y,
        cluster_info['offsets'],
        cluster_info['clusterB_idxs'],
        cluster_info['flat'],
        cluster_info['cluster_sizes'],
        n_triplets,
        rng
    )

@njit
def _sample_triplet(
    n,
    rng,
):
    if n < 3:
        raise ValueError("n must be at least 3")

    x_idx = rng.integers(0, n)

    y_idx = rng.integers(0, n - 1)
    if y_idx >= x_idx:
        y_idx += 1

    z_idx = rng.integers(0, n - 2)

    a = min(x_idx, y_idx)
    b = max(x_idx, y_idx)

    if z_idx >= a:
        z_idx += 1
    if z_idx >= b:
        z_idx += 1

    return x_idx, y_idx, z_idx

@njit
def _adi_kernel(
    X,
    Y,
    n_triplets: int,
    rng: np.random.Generator,
):
    sum_sqr = 0.0

    for _ in range(n_triplets):

        # sample triplet
        x_idx, y_idx, z_idx = _sample_triplet(
            X.shape[0],
            rng
        )

        # compute cosines
        cosX = _get_cosine(X, x_idx, y_idx, z_idx)
        cosY = _get_cosine(Y, x_idx, y_idx, z_idx)

        diff = cosX - cosY
        sum_sqr += diff * diff

    return sum_sqr / (4.0 * n_triplets)


@jaxtyped(typechecker=beartype)
def ADI(
    X: Real[np.ndarray, "N d"],
    Y: Real[np.ndarray, "N d2"],
    n_triplets: int = 0,
    random_seed: int | None | np.random.Generator = None,
) -> float:
    
    if not isinstance(random_seed, np.random.Generator):
        rng = np.random.default_rng(random_seed)
    else:
        rng = random_seed

    if n_triplets == 0:
        n_triplets = X.shape[0] * 10

    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    return _adi_kernel(
        X,
        Y,
        n_triplets=n_triplets,
        rng=rng
    )