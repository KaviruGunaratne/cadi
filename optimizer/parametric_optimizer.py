import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange
import matplotlib.pyplot as plt


# Determinism

def _set_determinism(seed: int | None):
    if seed is None:
        return

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class _TripletSampler:

    def __init__(self, labels: np.ndarray, seed: int | None):
        self.rng = np.random.default_rng(seed)

        classes, inverse = np.unique(labels, return_inverse=True)
        self.cluster_indices = []
        self.cluster_sizes = []

        for i in range(len(classes)):
            idx = np.where(inverse == i)[0]
            self.cluster_indices.append(idx)
            self.cluster_sizes.append(len(idx))

        self.cluster_sizes = np.array(self.cluster_sizes)
        self.n_clusters = len(self.cluster_sizes)

        if len(labels) < 3:
            raise ValueError("Need at least 3 points.")

        if self.n_clusters < 2:
            raise ValueError("Need at least 2 clusters.")

        self.valid_B = np.where(self.cluster_sizes >= 2)[0]
        if len(self.valid_B) == 0:
            raise ValueError("Need at least one cluster with >=2 points.")

    def sample(self, n_triplets: int):

        rng = self.rng

        cA = rng.integers(0, self.n_clusters, size=n_triplets)

        cB = self.valid_B[
            rng.integers(0, len(self.valid_B), size=n_triplets)
        ]

        mask_equal = cA == cB
        while np.any(mask_equal):
            cB[mask_equal] = self.valid_B[
                rng.integers(0, len(self.valid_B), size=np.sum(mask_equal))
            ]
            mask_equal = cA == cB

        x_idx = np.empty(n_triplets, dtype=np.int64)
        y_idx = np.empty(n_triplets, dtype=np.int64)
        z_idx = np.empty(n_triplets, dtype=np.int64)


        for c in np.unique(cA):
            mask = cA == c
            idx = self.cluster_indices[c]
            x_idx[mask] = idx[
                rng.integers(0, len(idx), size=np.sum(mask))
            ]

        for c in np.unique(cB):
            mask = cB == c
            idx = self.cluster_indices[c]
            size = len(idx)

            y_off = rng.integers(0, size, size=np.sum(mask))
            z_off = rng.integers(0, size, size=np.sum(mask))

            equal = y_off == z_off
            while np.any(equal):
                z_off[equal] = rng.integers(0, size, size=np.sum(equal))
                equal = y_off == z_off

            y_idx[mask] = idx[y_off]
            z_idx[mask] = idx[z_off]

        return x_idx, y_idx, z_idx


class _AngleMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


class AngleEmbedding:
    """
    Generates a 2D projection of a dataset, optimizing the Class Angular Distortion Index (CADI)
    Training is done by modelling a small MLP on this dataset

    Parameters
    ----------
    hidden_dim : int, default=128
        Dimensionality of the hidden layer of the MLP
    n_epochs : int, default=200
        Number of training epochs.
    triplets_per_epoch : int | None, default=None
        Number of random triplets sampled each epoch. If None, defaults to 10*N.
    lr : float, default=1e-2
        Initial learning rate for the Adam optimizer.
    seed : int | None, default=None
        RNG seed for reproducibility. If None, determinism is not enforced.
    device : str | None, default=None
        Torch device to use ('cuda' or 'cpu'). If None, chosen automatically.
    verbose : int, default=0
        Verbosity level
    """

    def __init__(
        self,
        hidden_dim=128,
        n_epochs=300,
        triplets_per_epoch=None,
        lr=1e-2,
        seed=None,
        device=None,
        compile_model=False,
        verbose=0,
    ):
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.triplets_per_epoch = triplets_per_epoch
        self.lr = lr
        self.seed = seed
        self.device = device
        self.compile_model = compile_model
        self.verbose = verbose


    def fit(self, X: np.ndarray, labels: np.ndarray):
        """
        Train the AngleEmbedding model on a dataset X.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
            Input data matrix of N samples with dimension d.
        labels : np.ndarray, shape (N,)
            Integer class labels used to sample triplets (anchor, positive,
            negative). Labels determine clusters from which triplet members
            are drawn.

        Returns
        -------
        self
            The fitted AngleEmbedding instance. The trained model is available
            as self.model and the embedding can be produced with transform().
        """

        _set_determinism(self.seed)

        if self.device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device

        X = X.astype(np.float32)
        N, d = X.shape

        if self.triplets_per_epoch is None:
            triplets_per_epoch = 10 * N
        else:
            triplets_per_epoch = self.triplets_per_epoch

        if self.verbose >= 2:
            print(f"[AngleEmbedding] Device: {device}")
            print(f"[AngleEmbedding] N={N}, d={d}")
            print(f"[AngleEmbedding] Triplets/epoch={triplets_per_epoch}")
            print(f"[AngleEmbedding] Epochs={self.n_epochs}")

        self.sampler = _TripletSampler(labels, self.seed)

        self.model = _AngleMLP(d, self.hidden_dim).to(device)


        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)


        X_torch = torch.from_numpy(X).to(device)

        if self.verbose >= 1:
            epoch_iter = trange(self.n_epochs, desc="Training")
        else:
            epoch_iter = range(self.n_epochs)


        for epoch in epoch_iter:

            x_idx, y_idx, z_idx = self.sampler.sample(triplets_per_epoch)

            x_idx = torch.from_numpy(x_idx).to(device)
            y_idx = torch.from_numpy(y_idx).to(device)
            z_idx = torch.from_numpy(z_idx).to(device)

            optimizer.zero_grad()

            Y_full = self.model(X_torch)

            v1X = X_torch[y_idx] - X_torch[x_idx]
            v2X = X_torch[z_idx] - X_torch[x_idx]

            v1Y = Y_full[y_idx] - Y_full[x_idx]
            v2Y = Y_full[z_idx] - Y_full[x_idx]

            cosX = F.cosine_similarity(v1X, v2X, dim=1)
            cosY = F.cosine_similarity(v1Y, v2Y, dim=1)


            loss = torch.mean((cosX - cosY) ** 2) / 4.0


            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0) # gradient clipping
            optimizer.step()
            scheduler.step()

            if self.verbose >= 2:
                epoch_iter.set_postfix(loss=float(loss.detach().cpu()), lr=optimizer.param_groups[0]['lr'])

            if self.verbose >= 100 and (epoch % 20 == 0):
                Y = Y_full.detach().cpu().numpy()
                plt.figure()
                plt.scatter(*Y.T, c=labels, s=1, cmap='rainbow')
                plt.title(f"Loss = {loss.item()}\nEpoch = {epoch}")
                plt.show()



        self.device_ = device
        self.X_ = X_torch

        if self.verbose >= 1:
            print("[AngleEmbedding] Training complete.")

        return self

    def transform(self, X: np.ndarray):
        """
        Embed input vectors using the trained model.

        Parameters
        ----------
        X : np.ndarray, shape (M, d)
            Data to embed. If M == N and the same data used for training,
            this returns the learned embeddings for those points.

        Returns
        -------
        Y : np.ndarray, shape (M, hidden_dim)
            Embeddings produced by the trained MLP.
        """
        X = X.astype(np.float32)
        X_torch = torch.from_numpy(X).to(self.device_)

        with torch.no_grad():
            Y = self.model(X_torch).cpu().numpy()
        return Y

    def fit_transform(self, X: np.ndarray, labels: np.ndarray):
        """
        Convenience method that fits the model and returns embeddings for X.

        Equivalent to calling fit(X, labels) followed by transform(X).

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
            Input data for training and embedding.
        labels : np.ndarray, shape (N,)
            Labels 

        Returns
        -------
        Y : np.ndarray, shape (N, hidden_dim)
            2D projection of X optimizing CADI
        """
        self.fit(X, labels)
        return self.transform(X)
