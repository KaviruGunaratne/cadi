import numpy as np

def labels_to_clusters_dict(labels: np.ndarray):
    classes = np.unique(labels)
    clusters_dict = {
        int(clz): np.flatnonzero(labels == clz)
        for clz in classes
    }

    return clusters_dict

def get_cluster_info(labels):
    classes = np.unique(labels)
    n_clusters = len(classes)

    if len(labels) < 3:
        raise ValueError("Dataset must be composed of at least 3 points")

    if n_clusters < 2:
        raise ValueError("Need at least 2 clusters to sample angles between clusters.")

    lengths = np.array([np.sum(labels == c) for c in classes])
    offsets = np.concatenate([[0], lengths[:-1].cumsum()])
    flat = np.concatenate([np.argwhere(labels == c).flatten() for c in classes])

    clusterB_idxs = np.argwhere(lengths >= 2).flatten()
    if not len(clusterB_idxs):
        raise ValueError("At least one cluster must have length >= 2 to sample y,z in triplet (x,y,z).")

    return {
        'flat': flat, 
        'offsets': offsets,
        'cluster_sizes': lengths,
        'clusterB_idxs': clusterB_idxs
    }