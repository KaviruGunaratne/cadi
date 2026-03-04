import numpy as np
from time import perf_counter
from collections import defaultdict
from tqdm import tqdm
import psutil

from metrics import CADI, cluster_distance_score
from metrics.utils import labels_to_clusters_dict
from zadu.measures import label_trustworthiness_and_continuity, steadiness_cohesiveness
from sklearn.metrics import silhouette_score, davies_bouldin_score, normalized_mutual_info_score, adjusted_rand_score
from hdbscan import HDBSCAN

RANDOM_SEED = 100

def cluster_with_hdbscan(X):
    return HDBSCAN(allow_single_cluster=False).fit_predict(X)



def find_time_taken(X, Y, labels, random_seed: int | None = None, num_runs=2):
    rng = np.random.default_rng(random_seed)

    cluster_metric_dict = {
        "CADI": lambda X, Y, labels, cluster_dict: CADI(X, Y, labels, n_triplets=X.shape[0] * 40, random_seed=random_seed),
        "CDS": lambda X, Y, labels, cluster_dict: cluster_distance_score(X, Y, cluster_dict),
        "SS": lambda X, Y, labels, cluster_dict: silhouette_score(Y, labels),
        "DBI": lambda X, Y, labels, cluster_dict:  davies_bouldin_score(Y, labels),
        "NMI": lambda X, Y, labels, cluster_dict: normalized_mutual_info_score(labels, cluster_with_hdbscan(Y)),
        "ARI": lambda X, Y, labels, cluster_dict: adjusted_rand_score(labels, cluster_with_hdbscan(Y)),
        "Label-T&C": lambda X, Y, labels, cluster_dict: label_trustworthiness_and_continuity.measure(X, Y, labels),
        "S&C": lambda X, Y, labels, cluster_dict: steadiness_cohesiveness.measure(X, Y),
        "CADI-100": lambda X, Y, labels, cluster_dict: CADI(X, Y, labels, n_triplets=X.shape[0] * 100, random_seed=random_seed),
    }
    
    list_of_results = list()
    if X.shape[0] < 1000:
        all_fracs = np.linspace(0.1, 1, 300)
    elif X.shape[0] < 50_000:
        all_fracs = np.linspace(0.01, 1, 300)
    else:
        all_fracs = np.linspace(0.001, 1, 300)
    actual_fracs = []
    
    for _ in range(num_runs):
        results = defaultdict(list)
        idxs_full = rng.permutation(X.shape[0])


        for frac in tqdm(all_fracs, desc='Fracs'):
            k = int(frac * X.shape[0])
            idxs = idxs_full[:k]

            sub_X = X[idxs]
            sub_Y = Y[idxs]
            sub_labels = labels[idxs]
            cluster_dict = labels_to_clusters_dict(sub_labels)
            
            with tqdm(total=len(cluster_metric_dict), desc=f"Metrics for frac = {frac}", leave=False) as metric_pbar:
                for metric_name, metric_func in cluster_metric_dict.items():
                    if sub_X.shape[0] > 15_000 and metric_name == "S&C":
                        metric_pbar.update(1)
                        continue


                    metric_pbar.set_postfix_str(metric_name)
                    start = perf_counter()
                    metric_func(sub_X, sub_Y, sub_labels, cluster_dict)
                    end = perf_counter()
                    results[metric_name].append(end - start)

                    metric_pbar.update(1)
            
            if not actual_fracs or frac != actual_fracs[-1]:
                actual_fracs.append(frac)

        list_of_results.append(results)
        

    return np.array(actual_fracs), list_of_results




if __name__ == "__main__":
    from project_consts import DATASET_DIR, LABELS_DIR, EMBEDDINGS_DIR, RESULTS_DIR
    from pathlib import Path
    import json
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        dsfiles = sorted(list(DATASET_DIR.iterdir()), reverse=True)
        for dsfile in tqdm(dsfiles):
            dataset = dsfile.stem
            # dataset = 'MNIST'

            X = np.load(DATASET_DIR / f"{dataset}.npy").astype(np.float64)
            labels = np.load(LABELS_DIR / f"{dataset}.npy")

            print(f"Dataset: {dataset} {X.shape}, {X.dtype}")
            print(f"Labels: {labels.shape}, {labels.dtype}")

            emb = 'TSNE'
            Y = np.load(EMBEDDINGS_DIR / dataset / f"{emb}.npy")

            res_dir = Path(RESULTS_DIR / f"time_taken_{dataset}_{emb}")
            if res_dir.exists():
                continue

            res_dir.mkdir(exist_ok=True, parents=True)

            fracs, results = find_time_taken(X, Y, labels, random_seed=RANDOM_SEED)
            
            frac_file = f"{dataset}_fracs.npy"
            results_file = f"{dataset}_time_taken.json"

            np.save(res_dir / frac_file, fracs)
            with open(res_dir / results_file, 'w') as f:
                json.dump(results, f, indent=4)