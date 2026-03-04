from pathlib import Path

DATA_DIR = Path('data')

DATASET_DIR = DATA_DIR / Path("datasets")
EMBEDDINGS_DIR =  DATA_DIR / Path("embeddings")
LABELS_DIR = DATA_DIR / Path("dataset_labels")

HDBSCAN_LABELS_DIR = DATA_DIR / Path("hdbscan_labels")
RND_TRIPLETS_DIR = DATA_DIR / Path("triplets")

RAW_DATA_DIR = Path('raw_data')

RESULTS_DIR = Path('results')

GENERAL_FIGS_DIR = Path("figs")
FIGS_DIR = GENERAL_FIGS_DIR / Path("emb_figs")

TEXT_DIR = DATA_DIR / "texts"