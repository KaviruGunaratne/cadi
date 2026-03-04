from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.neighbors import radius_neighbors_graph
import matplotlib.patches as mpatches

def estimate_epsilon(X, k=5, percentile=90):
    """
    `percentile`th percentile of `k`th nearest neighbors
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    kth_dist = distances[:, -1]
    return np.percentile(kth_dist, percentile)


def build_epsilon_graph(X, eps, weighted=True, symmetric=True):
    """
    Adjacency matrix of epsilon graph
    As sparse matrix
    """
    mode = 'distance' if weighted else 'connectivity'
    A = radius_neighbors_graph(X, radius=eps, mode=mode)

    if symmetric:
        A = 0.5 * (A + A.T)

    return A

def draw_projection_and_graph(A, Y, labels=None,
                              edge_cmap='plasma',
                              fig_title="",
                              label_map=None,
                              plot_args=None
):
    if plot_args is None:
        plot_args = {
            's': 1,
            'cmap': 'rainbow',
        }
    plot_args_wo_cmap = {k: v for k,v in plot_args.items() if k != 'cmap'}

    A = A.tocoo()
    rows, cols, weights = A.row, A.col, A.data

    fig = plt.figure(figsize=(9, 4))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])  # dedicated colorbar axis

    min_x, max_x = Y[:, 0].min(), Y[:, 0].max()
    min_y, max_y = Y[:, 1].min(), Y[:, 1].max()

    padding = 0.05
    span = max(max_x - min_x, max_y - min_y)
    span *= (1 + 2 * padding)

    mid_x = 0.5 * (min_x + max_x)
    mid_y = 0.5 * (min_y + max_y)

    for ax in (ax1, ax2):
        ax.set_xlim(mid_x - span/2, mid_x + span/2)
        ax.set_ylim(mid_y - span/2, mid_y + span/2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')


    
    # Left: projection
    if labels is None:
        ax1.scatter(*Y.T, **plot_args_wo_cmap)
    else:
        unique_labels = np.unique(labels)
        cmap = plt.colormaps[plot_args['cmap']]

        # Normalize label values to colormap range
        norm = plt.Normalize(vmin=unique_labels.min(),
                            vmax=unique_labels.max())

        # Scatter plot
        ax1.scatter(*Y.T, c=labels,
                    **plot_args,
        )

        # Create legend handles
        handles = []
        for lab in unique_labels:
            color = cmap(norm(lab))
            handles.append(
                mpatches.Patch(
                    color=color,
                    label=str(lab) if label_map is None else label_map[lab]
                )
            )

        ax1.legend(handles=handles,
                title="Labels",
                loc="best",
                frameon=True)


    ax1.set_title("Projection")


    # Right: graph
    ax2.scatter(*Y.T, s=plot_args['s'], alpha=0.5, c='black', zorder=0)

    if len(weights) > 0:
        norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())
        cmap = plt.colormaps[edge_cmap]
        colors = cmap(norm(weights))

        for i, j, color in zip(rows, cols, colors):
            if i < j:
                ax2.plot(
                    [Y[i, 0], Y[j, 0]],
                    [Y[i, 1], Y[j, 1]],
                    color=color,
                    alpha=0.6,
                    # alpha=np.mean(color),
                    linewidth=1,
                    zorder=1
                )

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        fig.colorbar(sm, cax=cax, label="Edge Weight (Distance)")

    else:
        cax.axis("off")  # hide empty colorbar axis

    ax2.set_title("Graph")

    fig.suptitle(fig_title)
    return fig
