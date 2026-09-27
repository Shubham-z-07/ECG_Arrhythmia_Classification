import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_confusion_matrix(
    matrix: pd.DataFrame,
    output_path,
    title: str,
    normalize_rows: bool = True,
) -> None:
    values = matrix.astype(float)
    fmt = "d"
    cmap_values = matrix
    if normalize_rows:
        values = values.div(values.sum(axis=1).replace(0, 1), axis=0)
        fmt = ".2f"
        cmap_values = values

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        cmap_values,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_metric_barplot(summary: pd.DataFrame, output_path) -> None:
    plot_df = summary.melt(
        id_vars=["model", "split"],
        value_vars=["accuracy", "macro_f1", "weighted_f1"],
        var_name="metric",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="model", y="score", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Classical model metrics")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
