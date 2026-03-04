import numpy as np

def make_one_doll(c, R, k, nx, nv):
    # Define circles (product of circles)
    def A(x): return (x + c)**2 - R**2
    def B(x): return (x - c)**2 - R**2

    # Compute radius r(x)
    def r_of_x(x):
        AA = A(x)
        BB = B(x)
        disc = (AA - BB)**2 + 4*k
        t_plus = (-(AA + BB) + np.sqrt(disc)) / 2
        t_plus = np.maximum(t_plus, 0)  # numerical safety
        return np.sqrt(t_plus)
    
    # Determine x-domain where r(x) > 0
    xs = np.linspace(-5, 5, 2000)
    rs = r_of_x(xs)
    mask = rs > 1e-6
    xmin = xs[mask][0]
    xmax = xs[mask][-1]

    # Build surface grid
    x_vals = np.linspace(xmin, xmax, nx + 2)[1:-1]
    v_vals = np.linspace(0, 2*np.pi, nv + 1)[:-1]
    X, V = np.meshgrid(x_vals, v_vals)
    Rvals = r_of_x(X)
    Y = Rvals * np.cos(V)
    Z = Rvals * np.sin(V)

    dataset = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    return dataset

def make_sphere(c, r, theta_pts, phi_pts):
    theta = np.linspace(0, 2 * np.pi, theta_pts + 1)[:-1]
    phi = np.linspace(0, np.pi, phi_pts + 1)[:-1]
    theta, phi = [angle.flatten() for angle in np.meshgrid(theta, phi)]

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    X = np.column_stack([x, y, z])

    X += c

    return X


def matryoshka():
    c = 2
    Rs = [1, 1.5, 2]
    k = 10

    rs = [0.3, 0.7]

    coords = []
    labels = []
    for i, R in enumerate(Rs):
        doll = make_one_doll(c, R, k, 40, 40)
        doll_lbls = np.full(doll.shape[0], i)
        coords.append(doll)
        labels.append(doll_lbls)


    for i, r in enumerate(rs):
        for side in [1, -1]:
            sphere = make_sphere(np.array([c, 0, 0]) * side, r, 20, 20)
            sphere_lbls = np.full(sphere.shape[0], (i + len(Rs) * side))
            coords.append(sphere)
            labels.append(sphere_lbls)

    X = np.concatenate(coords)
    labels = np.concatenate(labels)

    return X, labels