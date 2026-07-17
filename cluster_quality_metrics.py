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

import numpy as np
import matplotlib.pyplot as plt


def plot_heatmap_academic(
    T,
    save_path="heatmap_academic",
    x_label="Target Prototype",
    y_label="Source Prototype",
    title=None,
    cmap="BuGn"
):
    """
    绘制学术风格热力图：
    1. 不显示单元格数值
    2. 不显示图例(colorbar)
    3. 采用冷色调
    4. 保存为 svg / pdf / png

    Parameters
    ----------
    T : array-like, shape (n, m)
        输入矩阵
    save_path : str
        保存路径（不带后缀）
    x_label : str
        横轴标签
    y_label : str
        纵轴标签
    title : str or None
        标题，论文中通常可设为 None
    cmap : str
        冷色调配色，推荐:
        "BuGn", "GnBu", "PuBu", "Blues", "YlGnBu", "bone", "winter"
    """

    T = np.asarray(T, dtype=float)
    if T.ndim != 2:
        raise ValueError("T 必须是二维矩阵")

    n_rows, n_cols = T.shape

    # 学术风格参数
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    # 自适应尺寸
    fig_w = max(5, min(10, 0.6 * n_cols + 2))
    fig_h = max(4, min(9, 0.6 * n_rows + 1.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    # 绘制热力图
    im = ax.imshow(
        T,
        cmap=cmap,
        interpolation="nearest",
        aspect="equal"
    )

    # 坐标轴标签
    ax.set_xlabel(x_label, labelpad=8)
    ax.set_ylabel(y_label, labelpad=8)

    if title is not None:
        ax.set_title(title, pad=10)

    # 刻度
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels([rf"$p_{{{i+1}}}$" for i in range(n_cols)])
    ax.set_yticklabels([rf"$p_{{{i+1}}}$" for i in range(n_rows)])

    # 添加单元格边框线，使图更精致
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", linewidth=0.5, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3,
        width=0.8
    )

    # 不显示 colorbar（即去掉图例）
    # 故这里不添加 fig.colorbar(...)

    # 保证第一行在最上方
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    # 保存
    fig.savefig(f"{save_path}.svg", bbox_inches="tight", transparent=True)
    fig.savefig(f"{save_path}.pdf", bbox_inches="tight")
    fig.savefig(f"{save_path}.png", dpi=600, bbox_inches="tight")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # 示例矩阵
    T = np.array([
        [0.82, 0.07, 0.04, 0.07],
        [0.06, 0.78, 0.10, 0.06],
        [0.03, 0.09, 0.83, 0.05],
        [0.08, 0.04, 0.07, 0.81],
    ])

    plot_heatmap_academic(
        T,
        save_path="transition_heatmap",
        x_label="Destination Prototype",
        y_label="Starting Prototype",
        title=None,
        cmap="GnBu"   # 冷色调，推荐
    )
