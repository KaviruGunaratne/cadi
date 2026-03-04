from .concentric_spheres import concentric_spheres
from .rings import linked_line
from .nested_tori import nested_donuts
from .matryoshka import matryoshka

from project_consts import DATASET_DIR, LABELS_DIR
import numpy as np

def load_datasets():
    datasets = dict()

    # Rings
    n_rings = 20
    dim = 100
    ring_size = 200
    X, labels = linked_line(n_rings, dim, ring_size)
    rng = np.random.default_rng(42)
    X += rng.standard_normal(size=X.shape) * 2e-2 # Add noise
    datasets['rings'] = X, labels

    # 3D concentric spheres
    dim = 3
    n_spheres = 5
    points_per_sphere = 552
    datasets['concentric3'] = concentric_spheres(dim, n_spheres, points_per_sphere)

    # 4D concentric hyperspheres
    dim = 4
    n_spheres = 5
    points_per_sphere = 648
    datasets['concentric4'] = concentric_spheres(dim, n_spheres, points_per_sphere)

    # Nested donuts
    rs = [0.1, 1, 4]
    R = 16
    cross_pts = 25
    round_pts = 50
    datasets['donuts'] = nested_donuts(cross_pts, round_pts, R, rs)

    # Matryoshka
    datasets['matryoshka'] = matryoshka()

    return datasets
