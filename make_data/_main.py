from .real import load_datasets as load_real
from .synthetic import load_datasets as load_synthetic

import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE, MDS
from umap import UMAP
from optimizer import AngleEmbedding
from sklearn.decomposition import PCA
from pacmap import PaCMAP
from umato import UMATO
from scipy.spatial.distance import pdist, squareform

from project_consts import DATASET_DIR, LABELS_DIR, EMBEDDINGS_DIR
from metrics.utils import labels_to_clusters_dict

def save_datasets():
    DATASET_DIR.mkdir(exist_ok=True, parents=True)
    LABELS_DIR.mkdir(exist_ok=True, parents=True)


    datasets = load_synthetic() | load_real()

    for dataset_name, (X, labels) in datasets.items():
        file_name = f"{dataset_name}.npy"
        np.save(DATASET_DIR / file_name, X)
        np.save(LABELS_DIR / file_name, labels)


def save_embeddings(random_state=100):
    EMBEDDINGS_DIR.mkdir(exist_ok=True, parents=True)

    ds_files = list(DATASET_DIR.iterdir())

    def get_alg_funcs(labels):
        rng = np.random.default_rng(random_state)
        alg_funcs = {
            'TSNE': TSNE(n_components=2, random_state=random_state).fit_transform,
            'UMAP': UMAP(n_components=2, 
                        #  random_state=random_state
                        ).fit_transform,
            # 'AngleEmbedding': lambda X: angle_embed(X, perspective='clusters', cluster_dict=cluster_dict, seed=random_state),
            'AngleEmbedding': lambda X: AngleEmbedding(n_epochs=300, seed=random_state).fit_transform(X, labels),
            'PCA': PCA(n_components=2).fit_transform,
            # Because MDS has a bug, give precomputed distances
            # MDS sometimes internally creates an assymetrical distance matrix (due to numerical issues) and throws an error because of this
            # Error raised at 20 Newsgroups dataset
            # 'MDS': lambda X: MDS(n_init=1, init='random', n_components=2, random_state=random_state, dissimilarity='precomputed').fit_transform(squareform(pdist(X))),
            'MDS': MDS(n_init=1, init='random', random_state=random_state).fit_transform,
            'Random': lambda X: rng.random(size=(X.shape[0], 2)),
            'PaCMap': PaCMAP(random_state=random_state).fit_transform,
            'UMATO': UMATO(random_state=random_state).fit_transform,
        }
        return alg_funcs
    
    with tqdm(total=len(ds_files), position=0, desc='Dataset') as ds_pbar:
        for dsfile in ds_files:
            X = np.load(dsfile)
            dataset_name = dsfile.stem

            labels = np.load(LABELS_DIR / dsfile.name)

            ds_pbar.set_postfix_str(dataset_name)

            alg_funcs = get_alg_funcs(labels)

            with tqdm(total=len(alg_funcs), leave=False, position=1) as emb_pbar:
                embs = dict()
                for alg_name, alg_func in alg_funcs.items():
                    emb_pbar.set_postfix_str(alg_name)
                    Y = alg_func(X)
                    embs[alg_name] = Y
                    emb_pbar.update(1)

            embs_dir = (EMBEDDINGS_DIR / dataset_name)
            embs_dir.mkdir(exist_ok=True)
            for emb_name, emb in embs.items():
                np.save(embs_dir / f"{emb_name}.npy", emb)

            ds_pbar.update(1)