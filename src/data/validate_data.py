from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data.load_data import read_heartbeat_csv, split_features_labels
from src.visualization.eda_plots import (
    save_class_counts,
    save_examples_per_class,
    save_mean_waveforms,
)


def summarize_csv(path: Path) -> dict:
    df = read_heartbeat_csv(path)
    x, y = split_features_labels(df)
    unique_labels, counts = np.unique(y, return_counts=True)
    label_counts = {f"class_{int(k)}": int(v) for k, v in zip(unique_labels, counts)}

    summary = {
        "file": path.name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "feature_columns": int(x.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "min_label": int(y.min()),
        "max_label": int(y.max()),
        "n_classes": int(len(unique_labels)),
    }
    summary.update(label_counts)
    return summary


def build_class_distribution(y_train: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    labels = sorted(set(y_train.tolist()) | set(y_test.tolist()))
    rows = []
    for label in labels:
        train_count = int(np.sum(y_train == label))
        test_count = int(np.sum(y_test == label))
        rows.append(
            {
                "class": int(label),
                "train_count": train_count,
                "train_fraction": train_count / len(y_train),
                "test_count": test_count,
                "test_fraction": test_count / len(y_test),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_csvs = sorted(config.RAW_DATA_DIR.glob("*.csv"))
    if not raw_csvs:
        raise FileNotFoundError(f"No CSV files found in {config.RAW_DATA_DIR}")

    summary_df = pd.DataFrame([summarize_csv(path) for path in raw_csvs]).fillna(0)
    summary_path = config.TABLES_DIR / "raw_dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    train_df = read_heartbeat_csv(config.MITBIH_TRAIN_CSV)
    test_df = read_heartbeat_csv(config.MITBIH_TEST_CSV)
    x_train, y_train = split_features_labels(train_df)
    x_test, y_test = split_features_labels(test_df)

    class_dist = build_class_distribution(y_train, y_test)
    class_dist_path = config.TABLES_DIR / "mitbih_class_distribution.csv"
    class_dist.to_csv(class_dist_path, index=False)

    save_class_counts(
        y_train,
        y_test,
        config.FIGURES_DIR / "mitbih_class_counts.png",
    )
    save_examples_per_class(
        x_train,
        y_train,
        config.FIGURES_DIR / "mitbih_examples_per_class.png",
    )
    save_mean_waveforms(
        x_train,
        y_train,
        config.FIGURES_DIR / "mitbih_mean_waveforms.png",
    )

    print("Raw dataset summary")
    print(summary_df.to_string(index=False))
    print("\nMIT-BIH class distribution")
    print(class_dist.to_string(index=False))


if __name__ == "__main__":
    main()
