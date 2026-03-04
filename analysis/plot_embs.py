import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
import numpy as np

plot_args = {
    'olivetti': {'s': 25, 'cmap': 'tab20'},
    'penguins': {'s': 30},
    'coil20': {'s': 10},
    'matryoshka': {'s': 4, 'cmap': 'tab10', 'alpha':0.5},
    'fashionMNIST': {'s': 1, 'cmap': 'tab10'},
    'MNIST': {'cmap': 'tab10', 's': 1},
    'pendigits': {'s': 10, 'cmap': 'tab10'},
    'usps': {'s': 4, 'cmap': 'tab10'},
    'concentric4': {'s': 8},
    'concentric3': {'s': 8},
    'pbmc3k': {'s': 25, 'cmap': 'tab10'}, 
    'rings': {'s': 1, 'cmap': 'viridis'},
    'liver': {'s': 40, 'cmap': 'viridis'},
    'sentiment': {'s': 1, 'cmap': 'viridis'},
    'trec': {'s': 1, 'cmap': 'rainbow'},
    'emotion': {'s': 4, 'cmap': 'rainbow'}
}

def plot_emb(Y, labels, ax_len=5, fig_title="", ax = None, **scatter_args):
    if ax is None:
        fig, ax = plt.subplots(figsize=(ax_len, ax_len))
        fig.suptitle(fig_title)
    else:
        fig = None
        
    ax.scatter(*Y.T, c=labels, **scatter_args)
    ax.set_xticks([])
    ax.set_yticks([])
    # Determine square limits so aspect ratio is preserved
    min_x, max_x = Y[:,0].min(), Y[:,0].max()
    min_y, max_y = Y[:,1].min(), Y[:,1].max()
    padding = 0.05  # 5% padding
    span = max(max_x - min_x, max_y - min_y)
    span *= (1 + 2 * padding)
    mid_x = 0.5 * (min_x + max_x)
    mid_y = 0.5 * (min_y + max_y)

    ax.set_xlim(mid_x - span/2, mid_x + span/2)
    ax.set_ylim(mid_y - span/2, mid_y + span/2)
    ax.set_aspect("equal", adjustable="box")

    return fig


def plot_embs(embs, labels, plot_args, label_map=None, ncols=4, ax_len=4):
    nrows = len(embs) // ncols + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(ax_len * ncols, ax_len *nrows))
    axes = axes.flatten()

    if label_map:
        unique_labels = np.unique(labels)
        cmap = plt.get_cmap(plot_args['cmap'], len(unique_labels))
        color_dict = {
            label_map[lab]: cmap(i) for i, lab in enumerate(unique_labels)
        }

    for ax, (emb_name, Y) in zip(axes, embs.items()):
        ax.scatter(*Y.T, c=labels, **plot_args)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(emb_name)
    for ax in axes[len(embs):]:
        ax.remove()

    if label_map:
        legend_elements = [
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                label=str(lab),
                markerfacecolor=color_dict[lab],
                markersize=8
            )
            for lab in color_dict.keys()
        ]

        fig.legend(
            handles=legend_elements,
            loc="center left",
            bbox_to_anchor=(1.0, 0.7),
            title="Labels"
        )
    plt.show()