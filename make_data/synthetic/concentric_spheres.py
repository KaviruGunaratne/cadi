import numpy as np
from copy import copy

def make_n_sphere(n_points, dim):
    """
    Makes a (dim - 1)-sphere with n_points number of points
    Returns array with shape (n_points, dim)
    """
    if dim < 2:
        raise ValueError(f"dim was {dim} < 2. dim should be an int greater than or equal to 2.")

    n_per_angle = round(n_points ** (1 / (dim - 1)))
    n_phi = max(2, round(n_points / (n_per_angle ** (dim - 2)))) # More/Less for azimuthal angle


    thetas = [np.linspace(0, np.pi, n_per_angle + 2)[1:-1] for _ in range(dim - 2)]
    phi = np.linspace(0, 2 * np.pi, n_phi + 1)[:-1]
    angles = [angle.flatten() for angle in np.meshgrid(*thetas, phi)]

    coords = []
    x1 = np.ones_like(angles[0])
    x2 = np.ones_like(angles[0])
    for angle in angles[:-1]:
        x1 *= np.cos(angle)
        x2 *= np.sin(angle)
        coords.append(x1)
        x1 = copy(x2)

    coords.extend([x2 * np.cos(angles[-1]), x2 * np.sin(angles[-1])])

    X = np.column_stack(coords)

    # Enforce exactly n_points
    if X.shape[0] > n_points:
        idx = np.random.choice(X.shape[0], n_points, replace=False)
        X = X[idx]
    elif X.shape[0] < n_points:
        extra = np.random.normal(size=(n_points - X.shape[0], dim))
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        X = np.vstack([X, extra])

    return X

def concentric_spheres(dim, n_spheres, points_per_sphere):
    unit_sphere = make_n_sphere(points_per_sphere, dim)

    radii = [2 ** i for i in range(0, n_spheres)]
    spheres = [r * unit_sphere for r in radii]
    X = np.concatenate(spheres, axis=0)
    labels = np.concatenate([np.full(shape=points_per_sphere, fill_value=i) for i in range(len(radii))], axis=0)

    return X, labels