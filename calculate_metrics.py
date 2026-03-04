from project_consts import DATASET_DIR, LABELS_DIR, EMBEDDINGS_DIR, RESULTS_DIR
from metrics import CADI, cluster_distance_score
from metrics.utils import labels_to_clusters_dict
from zadu.measures import label_trustworthiness_and_continuity, steadiness_cohesiveness
from sklearn.metrics import silhouette_score, davies_bouldin_score, normalized_mutual_info_score, adjusted_rand_score

import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from typing import Literal
import warnings
from scipy.stats import hmean


RESULTS_FILE = RESULTS_DIR / "cluster_res.json"
rng = np.random.default_rng(42)
CLUSTER_STRATEGY: Literal["dbscan", "k-means"] = "dbscan"

def get_label_TnC(X, Y, labels):
    res_dict = label_trustworthiness_and_continuity.measure(X, Y, labels)
    lt = res_dict['label_trustworthiness']
    lc = res_dict['label_continuity']
    return hmean([lt, lc])

def get_SnC(X, Y):
    res_dict = steadiness_cohesiveness.measure(X, Y, clustering_strategy=CLUSTER_STRATEGY)
    s = res_dict['steadiness']
    c = res_dict['cohesiveness']
    return hmean([s, c])


def cluster_metrics(labels, random_seed=None):
    cluster_dict = labels_to_clusters_dict(labels)

    cluster_metric_dict = {
        'CADI': lambda X, Y: CADI(X, Y, labels, n_triplets=X.shape[0] * 10, random_seed=random_seed),
        "CDS": lambda X, Y: cluster_distance_score(X, Y, cluster_dict),
        'Label-T&C': lambda X, Y: get_label_TnC(X, Y, labels),
        "S&C": lambda X, Y: get_SnC(X, Y),
        'SS': lambda X, Y: silhouette_score(Y, labels),
        'DBI': lambda X, Y: davies_bouldin_score(Y, labels),
        "NMI": lambda X, Y: normalized_mutual_info_score(labels, cluster_with_hdbscan(Y)),
        "ARI": lambda X, Y: adjusted_rand_score(labels, cluster_with_hdbscan(Y)),
    }

    return cluster_metric_dict
    


def cluster_with_hdbscan(X):
    return HDBSCAN(allow_single_cluster=True).fit_predict(X)

def cluster_with_kmeans(X, k_clusters=100):
    return KMeans(n_clusters=k_clusters).fit_predict(X)

cluster_strat2func = {
    'dbscan': cluster_with_hdbscan,
    'k-means': cluster_with_kmeans,
}
cluster_func = cluster_strat2func[CLUSTER_STRATEGY]


def calc_and_save_metrics(RESULTS_FILE, rng):
    datasets = sorted(list(DATASET_DIR.iterdir()), reverse=True)
    results = defaultdict(dict)
    n_clusters_dict = dict()


    with tqdm(total=len(datasets), desc='Dataset', position=0) as ds_pbar:
        for dsfile in datasets:
            dataset_name = dsfile.name.replace(".npy", "")
            if dataset_name == 'sierpinski':
                continue
            
            X = np.load(dsfile)
            if not np.issubdtype(X.dtype, np.floating):
                X = X.astype(np.float64)
            
            # cluster_labels = cluster_func(X)
            cluster_labels = np.load(LABELS_DIR / dsfile.name)

            number_of_clusters = len(np.unique(cluster_labels))
            n_clusters_dict[dataset_name] = number_of_clusters
            # tqdm.write(f"{dataset_name} ({X.shape}): Identified {number_of_clusters} clusters")
            tqdm.write(f"Processing {dataset_name}")
            ds_pbar.set_postfix_str(f"{dataset_name} ({number_of_clusters})")
            
            metric_dict = cluster_metrics(cluster_labels, random_seed=rng)

            embfiles = list((EMBEDDINGS_DIR / dataset_name).iterdir())
            with tqdm(total=len(embfiles), position=1, leave=False) as emb_pbar:
                for emb_file in embfiles:
                    Y = np.load(emb_file)
                    emb_name = emb_file.name.replace(".npy", "")
                    emb_pbar.set_postfix_str(emb_name)

                    emb_results = dict()
                    with tqdm(total=len(metric_dict), position=2, leave=False) as metric_pbar:
                        for metric_name, res_func in metric_dict.items():
                            metric_pbar.set_postfix_str(metric_name)
                            emb_results[metric_name] = res_func(X, Y)

                            metric_pbar.update(1)

                    results[dataset_name][emb_name] = emb_results

                    emb_pbar.update(1)
            ds_pbar.update(1)

            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=4)

    return n_clusters_dict

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        print("Starting calculation of metrics...")
        n_clusters_dict = calc_and_save_metrics(RESULTS_FILE, rng)
        # print("Number of clusters identified in each dataset:")
        print("Number of classes per dataset")
        print(n_clusters_dict)


