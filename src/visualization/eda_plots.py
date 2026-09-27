import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def save_class_counts(y_train: np.ndarray, y_test: np.ndarray, output_path) -> None:
    labels = sorted(set(y_train.tolist()) | set(y_test.tolist()))
    train_counts = [int(np.sum(y_train == label)) for label in labels]
    test_counts = [int(np.sum(y_test == label)) for label in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2, train_counts, width, label="train")
    ax.bar(x + width / 2, test_counts, width, label="test")
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in labels])
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("MIT-BIH class distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_examples_per_class(
    x: np.ndarray,
    y: np.ndarray,
    output_path,
    examples_per_class: int = 4,
) -> None:
    rng = np.random.default_rng(42)
    labels = sorted(np.unique(y).tolist())
    fig, axes = plt.subplots(
        len(labels),
        examples_per_class,
        figsize=(3.2 * examples_per_class, 2.0 * len(labels)),
        sharex=True,
        sharey=True,
    )

    for row, label in enumerate(labels):
        idx = np.flatnonzero(y == label)
        chosen = rng.choice(idx, size=min(examples_per_class, len(idx)), replace=False)
        for col in range(examples_per_class):
            ax = axes[row, col]
            if col < len(chosen):
                ax.plot(x[chosen[col]], linewidth=1.2)
            ax.set_title(f"class {label}" if col == 0 else "")
            ax.grid(alpha=0.25)

    fig.suptitle("MIT-BIH example heartbeat segments", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_mean_waveforms(x: np.ndarray, y: np.ndarray, output_path) -> None:
    labels = sorted(np.unique(y).tolist())
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("tab10", n_colors=len(labels))

    for color, label in zip(palette, labels):
        class_x = x[y == label]
        mean_waveform = class_x.mean(axis=0)
        ax.plot(mean_waveform, label=f"class {label}", linewidth=1.8, color=color)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Amplitude")
    ax.set_title("MIT-BIH mean waveform by class")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
