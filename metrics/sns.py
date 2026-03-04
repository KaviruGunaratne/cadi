import numpy as np
from sklearn.metrics import pairwise_distances

def scale_normalized_stress(X, Y):
    """
    Compute scale-normalized stress between pairwise distances of X and Y.
    """
    dX = pairwise_distances(X)
    dY = pairwise_distances(Y)


    D_low_triu = dY[np.triu_indices(dY.shape[0], k=1)]
    D_high_triu = dX[np.triu_indices(dX.shape[0], k=1)]
    alpha = np.sum(D_low_triu * D_high_triu) / np.sum(np.square(D_low_triu))

    scaled_dY = alpha * dY

    diff_squared_sum = np.square(dX - scaled_dY).sum()
    orig_squared_sum = np.square(dX).sum()

    stress = np.sqrt(diff_squared_sum / orig_squared_sum)
    return stress