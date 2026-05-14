import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection


def plot_class_distribution(train_data, val_data, test_data,
                            class_names=None, save_path=None, show=True):
    splits = [("Train", train_data), ("Val", val_data), ("Test", test_data)]

    classes = sorted({int(c) for _, (_, y) in splits for c in np.unique(y)})
    counts = np.array([[int(np.sum(y == c)) for c in classes] for _, (_, y) in splits])
    pcts = counts / counts.sum(axis=1, keepdims=True) * 100

    n_classes = len(classes)
    bar_w = 0.8 / n_classes
    x = np.arange(len(splits))

    fig, ax = plt.subplots(figsize=(8, 5))
    class_names = class_names or {}
    for j, c in enumerate(classes):
        offsets = x + (j - (n_classes - 1) / 2) * bar_w
        label = class_names.get(c, f"class {c}")
        bars = ax.bar(offsets, counts[:, j], width=bar_w, label=label)
        for bar, cnt, pct in zip(bars, counts[:, j], pcts[:, j]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{cnt}\n({pct:.1f}%)",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in splits])
    ax.set_ylabel("# samples")
    ax.set_title("Class Distribution per Split")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.legend(title="label")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"saved figure → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_correlation_matrix(corr, title="Feature Correlation", colorbar_label="Correlation",
                            style="heatmap", save_path=None, show=True, max_ticks=30):
    values = corr.values if hasattr(corr, "values") else np.asarray(corr)
    n = values.shape[0]

    fig, ax = plt.subplots(figsize=(7, 6))

    if style == "heatmap":
        im = ax.imshow(values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    elif style == "ellipse":
        # r ≈ ±1 이면 길쭉한 선, r ≈ 0이면 원, 부호는 ±45° 기울기로 표현
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        positions = np.column_stack([jj.ravel(), ii.ravel()])
        r = values.ravel()
        widths = np.ones_like(r)
        heights = 1.0 - np.abs(r)
        angles = 45.0 * np.sign(r)
        im = EllipseCollection(
            widths=widths, heights=heights, angles=angles,
            units="x", offsets=positions, offset_transform=ax.transData,
            array=r, cmap="coolwarm", clim=(-1, 1),
        )
        im.set_edgecolor("black")
        im.set_linewidth(0.3)
        ax.add_collection(im)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_aspect("equal")
    else:
        raise ValueError(f"style must be 'heatmap' or 'ellipse', got {style!r}")

    fig.colorbar(im, ax=ax, label=colorbar_label)

    if n <= max_ticks:
        labels = list(corr.columns) if hasattr(corr, "columns") else list(range(n))
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=8)
    else:
        # feature가 많을 땐 tick label이 무의미해서 숨김
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title(f"{title} ({n}×{n})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"saved figure → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_normalization_comparison(x_before, x_after,
                                  n_samples=5, sample_indices=None,
                                  n_sensors=5, sensor_indices=None,
                                  bins=80, save_path=None, show=True):
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))

    axes[0, 0].hist(x_before.ravel(), bins=bins, color="C0", alpha=0.8)
    axes[0, 0].set_title("Distribution: Before")
    axes[0, 0].set_xlabel("value")
    axes[0, 0].set_ylabel("frequency")

    axes[0, 1].hist(x_after.ravel(), bins=bins, color="C1", alpha=0.8)
    axes[0, 1].set_title("Distribution: After")
    axes[0, 1].set_xlabel("value")

    if sample_indices is None:
        n_show = min(n_samples, x_before.shape[0])
        sample_indices = np.random.RandomState(0).choice(
            x_before.shape[0], n_show, replace=False
        )
    sample_indices = list(sample_indices)

    for i in sample_indices:
        axes[1, 0].plot(x_before[i], alpha=0.7, label=f"#{i}")
        axes[1, 1].plot(x_after[i], alpha=0.7, label=f"#{i}")
    axes[1, 0].set_title(f"Sample Traces (n={len(sample_indices)}): Before")
    axes[1, 0].set_xlabel("sensor index")
    axes[1, 0].set_ylabel("value")
    axes[1, 0].legend(fontsize=8, loc="upper right")
    axes[1, 1].set_title(f"Sample Traces (n={len(sample_indices)}): After")
    axes[1, 1].set_xlabel("sensor index")
    axes[1, 1].legend(fontsize=8, loc="upper right")

    if sensor_indices is None:
        n_show_sensors = min(n_sensors, x_before.shape[1])
        sensor_indices = np.random.RandomState(1).choice(
            x_before.shape[1], n_show_sensors, replace=False
        )
    sensor_indices = list(sensor_indices)

    for s in sensor_indices:
        axes[2, 0].plot(x_before[:, s], alpha=0.7, label=f"sensor_{s}")
        axes[2, 1].plot(x_after[:, s], alpha=0.7, label=f"sensor_{s}")
    axes[2, 0].set_title(f"Sensor Trajectories (n={len(sensor_indices)}): Before")
    axes[2, 0].set_xlabel("sample index")
    axes[2, 0].set_ylabel("value")
    axes[2, 0].legend(fontsize=8, loc="upper right")
    axes[2, 1].set_title(f"Sensor Trajectories (n={len(sensor_indices)}): After")
    axes[2, 1].set_xlabel("sample index")
    axes[2, 1].legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"saved figure → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None, show=True):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    n = cm.shape[0]

    classes = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
    if class_names:
        labels = [class_names.get(int(c), str(c)) for c in classes]
    else:
        labels = [str(c) for c in classes]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    cmap = plt.get_cmap("Blues")
    norm = plt.Normalize(vmin=cm.min(), vmax=cm.max())
    im = ax.imshow(cm, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(n):
        for j in range(n):
            r, g, b, _ = cmap(norm(cm[i, j]))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            color = "white" if luminance < 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"saved figure → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_roc_curve(y_true, y_score, save_path=None, show=True):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"saved figure → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_learning_curve(train_fn, predict_fn, x_train, y_train, x_valid, y_valid,
                        train_sizes=(0.1, 0.25, 0.5, 0.75, 1.0),
                        train_kwargs=None,
                        random_state=0, save_path=None, show=True):
    from sklearn.metrics import accuracy_score
    train_kwargs = train_kwargs or {}
    n = x_train.shape[0]
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(n)

    sizes, train_scores, val_scores = [], [], []
    for frac in train_sizes:
        size = max(1, int(n * frac))
        idx = indices[:size]
        x_sub = x_train[idx]
        y_sub = y_train[idx]
        m, _ = train_fn(x_sub, y_sub, x_valid, y_valid, **train_kwargs)
        train_scores.append(accuracy_score(y_sub, predict_fn(m, x_sub)))
        val_scores.append(accuracy_score(y_valid, predict_fn(m, x_valid)))
        sizes.append(size)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, train_scores, marker="o", label="train accuracy")
    ax.plot(sizes, val_scores, marker="s", label="valid accuracy")
    ax.set_xlabel("# training samples")
    ax.set_ylabel("accuracy")
    ax.set_title("Learning Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"saved figure → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_training_history(history, save_path=None, show=True):
    if history is None:
        print("plot_training_history: no per-epoch history available — skipped.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "accuracy" in history:
        axes[0].plot(history["accuracy"], label="train")
    if "val_accuracy" in history:
        axes[0].plot(history["val_accuracy"], label="valid")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    if "loss" in history:
        axes[1].plot(history["loss"], label="train")
    if "val_loss" in history:
        axes[1].plot(history["val_loss"], label="valid")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].set_title("Loss over Epochs")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"saved figure → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_topk_correlation(target_corrs, top_k=None, save_path=None, show=True):
    """각 feature와 target의 |correlation|을 내림차순 정렬해 bar chart로 표시.
    top_k가 주어지면 상위 K개를 다른 색으로 강조하고 cutoff line을 그림."""
    n = len(target_corrs)
    abs_corrs = np.abs(target_corrs)
    sorted_corrs = np.sort(abs_corrs)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))

    if top_k is not None and 0 < top_k < n:
        colors = ["C3" if i < top_k else "C0" for i in range(n)]
        cutoff = sorted_corrs[top_k - 1]
    else:
        colors = "C0"
        cutoff = None

    ax.bar(np.arange(n), sorted_corrs, color=colors, width=1.0)
    ax.set_xlabel("Rank (sorted by |correlation|)")
    ax.set_ylabel("|correlation with target|")
    ax.set_title(f"Feature ↔ Target Correlation ({n} features)")

    if cutoff is not None:
        ax.axhline(y=cutoff, color="red", linestyle="--", alpha=0.5,
                   label=f"top-{top_k} cutoff ({cutoff:.3f})")
        ax.legend(loc="upper right")

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"saved figure → {save_path}")
    if show:
        plt.show()
    plt.close(fig)
