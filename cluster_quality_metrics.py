import numpy as np
import torch
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_sc_score(features, labels):
    """
    Compute Silhouette Coefficient.

    Args:
        features: array-like or torch.Tensor, shape [N, D]
        labels: array-like or torch.Tensor, shape [N]

    Returns:
        float
    """
    X = _to_numpy(features)
    y = _to_numpy(labels).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"features must have shape [N, D], got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"labels must have shape [N], got {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("features and labels must have the same number of samples")

    unique_labels = np.unique(y)
    if unique_labels.shape[0] < 2:
        raise ValueError("Silhouette Coefficient requires at least 2 clusters")

    return float(silhouette_score(X, y))


def compute_db_index(features, labels):
    """
    Compute Davies-Bouldin Index.

    Args:
        features: array-like or torch.Tensor, shape [N, D]
        labels: array-like or torch.Tensor, shape [N]

    Returns:
        float
    """
    X = _to_numpy(features)
    y = _to_numpy(labels).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"features must have shape [N, D], got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"labels must have shape [N], got {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("features and labels must have the same number of samples")

    unique_labels = np.unique(y)
    if unique_labels.shape[0] < 2:
        raise ValueError("Davies-Bouldin Index requires at least 2 clusters")

    return float(davies_bouldin_score(X, y))


def plot_tsne(features, labels, title="t-SNE Visualization", random_state=42, savefilename = "tsne_clustering.png", Norm = True, dataname = 'None'):
    """
    Perform t-SNE and visualize clustering results.

    Args:
        features: array-like or torch.Tensor, shape [N, D]
        labels: array-like or torch.Tensor, shape [N]
        title: figure title
        random_state: random seed for t-SNE

    Returns:
        matplotlib.figure.Figure
    """
    X = _to_numpy(features)
    y = _to_numpy(labels).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"features must have shape [N, D], got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"labels must have shape [N], got {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("features and labels must have the same number of samples")

    tsne = TSNE(n_components=2, random_state=random_state)
    if Norm:
        X_2d = tsne.fit_transform(X)
    else:
        X_2d = X

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=10, alpha=0.8)
    ax.set_title(title)

    # ax.set_xlabel("t-SNE Dim 1")
    # ax.set_ylabel("t-SNE Dim 2")

    # legend = ax.legend(*scatter.legend_elements(), title="Clusters", loc="best")
    # ax.add_artist(legend)
    savefilename = dataname + '_' + savefilename
    fig.tight_layout()
    fig.savefig(savefilename, dpi=300, bbox_inches="tight")
    return fig
