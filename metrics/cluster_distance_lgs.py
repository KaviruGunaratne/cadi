import numpy as np
from jaxtyping import Real, Float, Int, jaxtyped
from typing import Dict, Any
from beartype import beartype
from sklearn.metrics import pairwise_distances

@jaxtyped(typechecker=beartype)
def normalized_stress(
    dX: Float[np.ndarray, "K K"],
    dY: Float[np.ndarray, "K K"],
    alpha: float | np.floating,
) -> float:  
    """
    Compute normalized stress between X and alpha*Y using zadu's stress measure.
    """
    dY = dY * alpha

    diff_sqr = np.square(dX - dY).sum()
    norm = np.square(dX).sum()
    stress = np.sqrt(diff_sqr / norm)
    return float(stress)

@jaxtyped(typechecker=beartype)
def SNS(
    dX: Float[np.ndarray, "K K"],
    dY: Float[np.ndarray, "K K"],
 ) -> float:
    D_low_triu = dY[np.triu_indices(dY.shape[0], k=1)]
    D_high_triu = dX[np.triu_indices(dX.shape[0], k=1)]
    alpha = np.sum(D_low_triu * D_high_triu) / np.sum(np.square(D_low_triu))
    return normalized_stress(dX, dY, alpha)

def cluster_distance_score(
    X: Real[np.ndarray, "N d"],
    Y: Real[np.ndarray, "N d2"],
    clusters: Dict[Any, Int[np.ndarray, "indexes"]],
    verbose = 0,
) -> float:
    # where each entry is the centroid of the corresponding cluster
    cluster_X = np.array([X[idxs].mean(axis=0) for idxs in clusters.values()])
    cluster_Y = np.array([Y[idxs].mean(axis=0) for idxs in clusters.values()])

    if verbose >= 10:
        print("Cluster centroids:")
        print("-----------------")
        print(cluster_X)
        print(cluster_Y)

    dX = pairwise_distances(cluster_X)
    dY = pairwise_distances(cluster_Y)

    if verbose >= 15:
        print("Centroid distance matrices")
        print("--------------------------")
        print(dX)
        print(dY)

    return SNS(dX, dY)