import numpy as np
from itertools import cycle

def linked_line(n_rings: int, dim: int, ring_size: int):
    X = np.zeros(shape=(n_rings, ring_size, dim))
    labels = np.empty(shape=(n_rings * ring_size))
    t_ring = np.linspace(0, 2 * np.pi, ring_size)
    for ring_idx, j in zip(range(n_rings), cycle(range(1, dim))):
        ring = X[ring_idx]
        ring[:, 0] = ring_idx * 1.5 + np.cos(t_ring)
        ring[:, j] = np.sin(t_ring)

        labels[ring_size * ring_idx: ring_size * (ring_idx + 1)] = ring_idx

    X = X.reshape(n_rings * ring_size, dim)

    return X, labels

def disconnected_line(n_rings: int, dim: int, ring_size: int):
    X = np.zeros(shape=(n_rings, ring_size, dim))
    labels = np.empty(shape=(n_rings * ring_size))
    t_ring = np.linspace(0, 2 * np.pi, ring_size)
    for ring_idx, j in zip(range(n_rings), cycle(range(1, dim))):
        ring = X[ring_idx]
        ring[:, 0] = ring_idx * 3 + np.cos(t_ring)
        ring[:, j] = np.sin(t_ring)

        labels[ring_size * ring_idx: ring_size * (ring_idx + 1)] = ring_idx

    X = X.reshape(n_rings * ring_size, dim)

    return X, labels

