from sklearn.datasets import make_blobs
import numpy as np
from sklearn.manifold import TSNE


def cluster_merge(embedding, labels, severity, n_clusters, random_state=42):
    """
    Merges a select number of clusters by translating them so that they overlap

    Parameters
    ----------

    severity: float, between 0.0 and 1.0
        How merged the clusters should be. 1.0 results in total overlap of the cluster centroids.

    n_pairs: int
        The number of clusters to overlap

    """
    severity **= 0.1
    
    rng = np.random.default_rng(random_state)
    emb = embedding.copy()
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    centroids = np.array([emb[labels == l].mean(axis=0) for l in unique_labels])

    # choose random pairs of distinct clusters
    cluster_choices = []
    if n_clusters >= 2:
        cluster_choices = rng.choice(n_clusters - 1, size=(n_clusters))

    out = emb.copy()

    i1 = n_clusters - 1
    c1 = centroids[i1]
    for i2 in cluster_choices:
        c2 = centroids[i2]
        # move cluster c2 toward cluster c1 by severity
        shift = (c1 - c2) * severity
        out[labels == unique_labels[i2]] = emb[labels == unique_labels[i2]] + shift

    return out

def cluster_shuffle(embedding, labels, severity=0.1):
    """Swap a fraction of points between clusters."""
    emb = embedding.copy()
    shuffled = emb.copy()
    n = len(labels)
    num_swaps = int(severity * n)

    for _ in range(num_swaps):
        i, j = np.random.choice(n, 2, replace=False)
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    return shuffled

def cluster_dispersion(embedding, labels, severity=0.2):
    """Expand each cluster outward from its centroid."""
    severity = severity * 10
    emb = embedding.copy()
    distorted = np.zeros_like(emb)
    unique_labels = np.unique(labels)
    for l in unique_labels:
        cluster_points = emb[labels == l]
        centroid = cluster_points.mean(axis=0)
        distorted[labels == l] = centroid + (cluster_points - centroid) * (1 + severity)
    return distorted

def generate():
    X, labels = make_blobs(1000, 10, centers=5)
    y_correct = TSNE().fit_transform(X)

    args = {
        "dispersion": [
            {"severity": 0.2},
            {"severity": 0.3},
            {"severity": 2}
        ],
        "merge": [
            {"severity": 1, "n_pairs": 1},
            {"severity": 1, "n_pairs": 2},
            {"severity": 1, "n_pairs": 3}
        ],
        "shuffle": [
            {"severity": 0.05},
            {"severity": 0.2},
            {"severity": 0.7}
        ]
    }
    levels = ['bad', 'worse', 'worst']
    args = {key: dict(zip(levels, value)) for key, value in args.items()}


    """
    keys: merge, shuffle, dispersion
    values:
        ({'dataset': X, embeddings_dict}, labels)
    """
    out = dict()

    for distortion_name, func in {
        "merge": cluster_merge,
        "shuffle": cluster_shuffle,
        "dispersion": cluster_dispersion,
    }.items():
        embeddings = {level: func(y_correct, labels, **args[distortion_name][level]) for level in levels}

        out |= {
            distortion_name: ({'dataset': X} | embeddings, labels)
        }

    return out