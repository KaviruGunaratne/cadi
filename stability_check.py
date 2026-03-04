from metrics import CADI
from project_consts import DATASET_DIR, LABELS_DIR, EMBEDDINGS_DIR
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

if __name__ == '__main__':
    dsfiles = list(reversed(list(DATASET_DIR.iterdir())))
    print([dsfile.stem for dsfile in dsfiles])
    with tqdm(total=len(dsfiles), desc="Dataset", position=0) as ds_pbar:
        for dsfile in dsfiles:
            dataset = dsfile.stem
            ds_pbar.set_postfix_str(dataset)

            X = np.load(dsfile)
            labels = np.load(LABELS_DIR / dsfile.name)
            embfile = EMBEDDINGS_DIR / dataset / "TSNE.npy"
            if not embfile.exists():
                ds_pbar.update(1)
                continue
            Y = np.load(embfile)
 
            # k_vals = np.arange(2, 21, 2)
            k_vals = [1] + list(range(5, 41, 5))
            n_runs = 10_000

            def compute_score(X, Y, labels, k):
                return CADI(X, Y, labels, n_triplets=int(k * X.shape[0]))

            all_scores = dict()
            for k in tqdm(k_vals, desc="k values", position=1, leave=False):
                all_scores[int(k)] = Parallel(n_jobs=-1)(
                    delayed(compute_score)(X, Y, labels, k) 
                    for _ in tqdm(range(n_runs), desc=f"k={k}", position=2, leave=False)
                )

            from project_consts import RESULTS_DIR
            import json
            out_dir = RESULTS_DIR / 'stability5'
            out_dir.mkdir(exist_ok=True)
            with open(out_dir / f"{dataset}_stability.json", 'w') as f:
                json.dump(all_scores, f, indent=4)

            ds_pbar.update(1)