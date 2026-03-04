import numpy as np

def make_donut(cross_pts, round_pts, R, r):
    t = np.linspace(0, 2 * np.pi, cross_pts + 1)[:-1]
    v = np.tile(t, round_pts)
    t2 = np.linspace(0, 2 * np.pi, round_pts + 1)[:-1]
    u = np.repeat(t2, cross_pts)

    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    torus = np.column_stack([x, y, z])
    return torus

def nested_donuts(cross_pts, round_pts, R, rs):
    labels = []
    coords = []

    for i, r in enumerate(rs):
        torus = make_donut(cross_pts, round_pts, R, r)
        tor_lbls = np.full(torus.shape[0], i)
        labels.append(tor_lbls)
        coords.append(torus)

    labels = np.concatenate(labels, axis=0)
    X = np.concatenate(coords, axis=0)

    return X, labels