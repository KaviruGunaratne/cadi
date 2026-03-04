import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces, fetch_openml, fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import seaborn as sns
from tqdm import tqdm
import urllib.request
import zipfile
from PIL import Image
import io
import pandas as pd
import gzip
import struct
import tempfile
import tarfile
import requests

LIVER_LABELLING = ['HCC', 'normal']
TREC_LABELLING = ['ABBR', 'DESC', 'ENTY', 'HUM', 'LOC', 'NUM']
AG_NEWS_LABELLING = ['World', 'Sports', 'Business','Sci/Tech']
EMOTION_LABEL_MAP = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def save_texts(dataset_name, texts):
    from project_consts import TEXT_DIR
    import json
    TEXT_DIR.mkdir(exist_ok=True, parents=True)
    with open(TEXT_DIR / f"{dataset_name}.json", 'w') as f:
        json.dump(texts, f)


def get_olivetti():
    ol = fetch_olivetti_faces()
    return ol.data, ol.target

def get_coil20():
    url = "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"
    
    # Download ZIP into memory
    print(f"Downloading {url} ...")
    resp = urllib.request.urlopen(url)
    data = resp.read()  # bytes
    
    # Open ZIP from memory
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        X_list, labels_list = [], []
        
        # Only PNGs inside coil-20-proc/
        for name in sorted(zf.namelist()):
            if not name.lower().endswith(".png"):
                continue
            
            # Read PNG file bytes into memory
            with zf.open(name) as file:
                img = Image.open(file).convert("L")
                img_vec = np.array(img).flatten()
                X_list.append(img_vec)
                
                # Extract label from filename
                fname = os.path.basename(name)
                objpart = fname.split("__")[0]
                label = int(objpart.replace("obj", "")) - 1
                labels_list.append(label)
    
    X = np.array(X_list)
    labels = np.array(labels_list)
    return X, labels

def get_coil100():
    url = "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip"

    # Download ZIP into memory
    print(f"Downloading {url} ...")
    data = urllib.request.urlopen(url).read()

    # Open ZIP from memory
    X_list, labels_list = [], []

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in sorted(zf.namelist()):
            if not name.lower().endswith(".png"):
                continue

            # Load image directly from ZIP (no disk)
            with zf.open(name) as f:
                img = Image.open(f).convert("L")
                arr = np.array(img).flatten()
                X_list.append(arr)

            # Extract label
            fname = os.path.basename(name)
            obj = fname.split('__')[0]       # e.g., "obj23"
            label = int(obj.replace("obj", "")) - 1
            labels_list.append(label)

    return np.array(X_list), np.array(labels_list)

def get_pendigits():
    pen = fetch_openml(name="pendigits", version=1, as_frame=False)
    return pen.data, pen.target.astype(np.float16)

def get_usps():
    usps = fetch_openml(name="usps", version=2, as_frame=False)
    return usps.data, usps.target.astype(np.float16)

def get_20newsgroups():
    train = fetch_20newsgroups(subset="train", remove=('headers', 'footers', 'quotes'))
    test = fetch_20newsgroups(subset="test", remove=('headers', 'footers', 'quotes'))
    
    texts = train.data + test.data
    labels = np.concatenate([train.target, test.target])

    # Generate embeddings
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    return embeddings, labels

def get_emotion(return_texts=False):
    ds_cfg = {
        "name": "emotion",
        "split": "train",
        "text_field": "text",
        "label_field": "label",
    }

    model = SentenceTransformer(EMBEDDING_MODEL)

    dataset = load_dataset(ds_cfg["name"], split=ds_cfg["split"])

    texts = dataset[ds_cfg["text_field"]]
    labels = np.array(dataset[ds_cfg["label_field"]])

    # Compute embeddings
    X = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    save_texts('emotion', list(texts))
    
    if return_texts:
        return X, labels, texts
    else:
        return X, labels

def get_penguins():
    data = sns.load_dataset('penguins').dropna(thresh=6)
    cols_num = ['bill_length_mm', 'bill_depth_mm',
                'flipper_length_mm', 'body_mass_g']
    X = data[cols_num]
    species_mapping = {species: idx for idx, species in enumerate(data['species'].unique())}
    labels = data['species'].map(species_mapping).to_numpy()

    X = MinMaxScaler().fit_transform(X)
    
    return X, labels

def get_fashion_mnist():
    """
    Download and return the Fashion-MNIST training set as (X, labels),
    with no persistent storage. X has shape (60000, 784).
    """
    BASE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    img_url = BASE + "train-images-idx3-ubyte.gz"
    lab_url = BASE + "train-labels-idx1-ubyte.gz"

    # download to temporary files
    img_tmp = tempfile.mkstemp()[1]
    lab_tmp = tempfile.mkstemp()[1]
    urllib.request.urlretrieve(img_url, img_tmp)
    urllib.request.urlretrieve(lab_url, lab_tmp)

    # decompress into memory
    with gzip.open(img_tmp, 'rb') as f:
        img_raw = f.read()
    with gzip.open(lab_tmp, 'rb') as f:
        lab_raw = f.read()

    # clean up temporary files
    os.remove(img_tmp)
    os.remove(lab_tmp)

    # parse IDX images
    magic, num, rows, cols = struct.unpack(">IIII", img_raw[:16])
    X = np.frombuffer(img_raw[16:], dtype=np.uint8).reshape(num, rows * cols)

    # parse IDX labels
    magic, num = struct.unpack(">II", lab_raw[:8])
    labels = np.frombuffer(lab_raw[8:], dtype=np.uint8)

    return X, labels

def get_mnist():
    BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    img_url = BASE + "train-images-idx3-ubyte.gz"
    lab_url = BASE + "train-labels-idx1-ubyte.gz"

    img_tmp = tempfile.mkstemp()[1]
    lab_tmp = tempfile.mkstemp()[1]
    urllib.request.urlretrieve(img_url, img_tmp)
    urllib.request.urlretrieve(lab_url, lab_tmp)

    with gzip.open(img_tmp, 'rb') as f:
        img_raw = f.read()
    with gzip.open(lab_tmp, 'rb') as f:
        lab_raw = f.read()

    os.remove(img_tmp)
    os.remove(lab_tmp)

    magic, num, rows, cols = struct.unpack(">IIII", img_raw[:16])
    X = np.frombuffer(img_raw[16:], dtype=np.uint8).reshape(num, rows * cols)

    magic, num = struct.unpack(">II", lab_raw[:8])
    labels = np.frombuffer(lab_raw[8:], dtype=np.uint8)

    return X, labels

def get_pbmc3k():
    import scanpy as sc
    adata = sc.datasets.pbmc3k()

    # Preprocessing
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sc.pp.scale(adata, max_value=10)

    # PCA
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

    # Clustering
    sc.tl.leiden(adata, resolution=0.5, flavor='igraph')

    X = adata.X
    labels = adata.obs['leiden'].to_numpy(dtype=np.int32)

    return X, labels


def get_liver():
    url = "https://sbcb.inf.ufrgs.br/data/cumida/Genes/Liver/GSE14520_U133A/Liver_GSE14520_U133A.csv"
    df = pd.read_csv(url, index_col=0)

    lbl_col = 'type'
    labels = df[lbl_col].values
    for i, lbl in enumerate(LIVER_LABELLING):
        labels[labels == lbl] = i
    labels = labels.astype(int)

    X = df.drop(columns=[lbl_col]).values
    # PCA Prepocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=50, random_state=0)
    X = pca.fit_transform(X_scaled)

    return X, labels


def get_uci_sentiment(return_texts=False):
    """
    Loads and merges Amazon, IMDb, and Yelp sentiment datasets.
    
    Returns:
        X      : np.ndarray of shape (N, embedding_dim)
        y      : np.ndarray of shape (N,)
        texts  : list[str]
    """
    UCI_SENTIMENT_ZIP_URL = (
        "https://archive.ics.uci.edu/static/public/331/"
        "sentiment+labelled+sentences.zip"
    )


    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = []
    labels = []

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "sentiment.zip")

        # Download zip
        r = requests.get(UCI_SENTIMENT_ZIP_URL)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(r.content)

        # Extract zip
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        base_path = os.path.join(
            tmpdir,
            "sentiment labelled sentences"
        )

        files = [
            "amazon_cells_labelled.txt",
            "imdb_labelled.txt",
            "yelp_labelled.txt",
        ]

        # Load and merge
        for fname in files:
            file_path = os.path.join(base_path, fname)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    sentence, label = line.strip().rsplit("\t", 1)
                    texts.append(sentence)
                    labels.append(int(label))

        # Compute embeddings
        embeddings = []
        for text in tqdm(texts, desc="Encoding UCI Sentiment"):
            emb = model.encode(text, show_progress_bar=False)
            embeddings.append(emb)

        X = np.vstack(embeddings)
        y = np.array(labels)

    save_texts('sentiment', list(texts))

    if return_texts:
        return X, y, texts
    else:
        return X, y

def get_ag_news(return_texts=False):
    """
    Loads AG News dataset (title only).

    Returns:
        X      : np.ndarray of shape (N, embedding_dim)
        y      : np.ndarray of shape (N,)
        texts  : list[str]
        label_map : dict[str, int]
    """

    model = SentenceTransformer(EMBEDDING_MODEL)

    dataset = load_dataset("ag_news", split="train")

    print(dataset.features)

    texts = [item["text"] for item in dataset]
    labels = [item["label"] for item in dataset]

    y = np.array(labels)

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    X = np.vstack(embeddings)

    save_texts('ag_news', list(texts))

    if return_texts:
        return X, y, texts
    else:
        return X, y
    
def get_trec(return_texts=False):
    """
    Loads and merges all 5 TREC train sets (coarse labels).

    Returns:
        X      : np.ndarray of shape (N, embedding_dim)
        y      : np.ndarray of shape (N,)
        texts  : list[str]
        label_map : dict[str, int]
    """
    BASE_URL = "https://cogcomp.seas.upenn.edu/Data/QA/QC/"
    TRAIN_FILES = [
        "train_1000.label",
        "train_2000.label",
        "train_3000.label",
        "train_4000.label",
        "train_5500.label",
    ]

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = []
    coarse_labels = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for fname in TRAIN_FILES:
            url = BASE_URL + fname
            local_path = os.path.join(tmpdir, fname)

            r = requests.get(url)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)

            with open(local_path, "r", encoding="latin-1") as f:
                for line in f:
                    label_part, question = line.strip().split(" ", 1)
                    coarse = label_part.split(":")[0]
                    texts.append(question)
                    coarse_labels.append(coarse)

        label_map = {label: i for i, label in enumerate(TREC_LABELLING)}

        y = np.array([label_map[l] for l in coarse_labels])

        # Compute embeddings
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        X = np.vstack(embeddings)
        save_texts('trec', list(texts))

    if return_texts:
        return X, y, texts
    else:
        return X, y


def get_acl_imdb(return_texts=False):
    """
    Downloads the ACL IMDB dataset, reads all pos/neg train + test
    reviews, and returns embeddings, labels, and raw texts.

    Returns:
        X      : np.ndarray of shape (N, embedding_dim)
        y      : np.ndarray of shape (N,)
        texts  : list[str]
    """
    IMDB_TAR_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = []
    labels = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, "aclImdb_v1.tar.gz")

        # Download tarball
        r = requests.get(IMDB_TAR_URL, stream=True)
        r.raise_for_status()
        with open(tar_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmpdir)

        base = os.path.join(tmpdir, "aclImdb")

        # Define paths to labeled dirs
        labeled_dirs = [
            os.path.join(base, "train", "pos"),
            os.path.join(base, "train", "neg"),
            os.path.join(base, "test", "pos"),
            os.path.join(base, "test", "neg"),
        ]

        # Label mapping: pos=1, neg=0
        for dir_path in labeled_dirs:
            is_pos = "pos" in dir_path.split(os.sep)[-1]
            label = 1 if is_pos else 0

            for fname in os.listdir(dir_path):
                file_path = os.path.join(dir_path, fname)
                if not os.path.isfile(file_path):
                    continue
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if text:
                        texts.append(text)
                        labels.append(label)

        # Compute embeddings
        X = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        y = np.array(labels)

    save_texts('acl_imdb', list(texts))

    if return_texts:
        return X, y, texts
    else:
        return X, y

DATASET_LOADERS = datasets = {
    # "20newsgroups": get_20newsgroups,
    "acl_imdb": get_acl_imdb,
    "ag_news": get_ag_news,
    "coil20": get_coil20,
    "coil100": get_coil100,
    "emotion": get_emotion,
    "fashionMNIST": get_fashion_mnist,
    "liver": get_liver,
    "MNIST": get_mnist,
    "olivetti": get_olivetti,
    "pbmc3k": get_pbmc3k,
    "pendigits": get_pendigits,
    "penguins": get_penguins,
    "sentiment": get_uci_sentiment,
    'trec': get_trec,
    "usps": get_usps,
}

def load_datasets():
    return {dataset_name: func() for dataset_name, func in tqdm(DATASET_LOADERS.items())}