import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.data.load_data import load_mitbih


def per_beat_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize each heartbeat independently."""
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return ((x - mean) / np.maximum(std, eps)).astype(np.float32)


def describe_waveforms(name: str, x: np.ndarray) -> dict:
    row_std = x.std(axis=1)
    return {
        "split": name,
        "rows": int(x.shape[0]),
        "time_steps": int(x.shape[1]),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median_row_std": float(np.median(row_std)),
        "zero_fraction": float(np.mean(x == 0.0)),
    }


def class_distribution(y: np.ndarray) -> dict[int, int]:
    labels, counts = np.unique(y, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def main() -> None:
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    x_train_full, y_train_full, x_test, y_test = load_mitbih()
    all_indices = np.arange(len(y_train_full))

    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=config.VALIDATION_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y_train_full,
    )

    x_train = x_train_full[train_idx]
    y_train = y_train_full[train_idx]
    x_val = x_train_full[val_idx]
    y_val = y_train_full[val_idx]

    np.save(config.X_TRAIN_RAW_NPY, x_train)
    np.save(config.Y_TRAIN_NPY, y_train)
    np.save(config.X_VAL_RAW_NPY, x_val)
    np.save(config.Y_VAL_NPY, y_val)
    np.save(config.X_TEST_RAW_NPY, x_test)
    np.save(config.Y_TEST_NPY, y_test)

    np.save(config.X_TRAIN_ZSCORE_NPY, per_beat_zscore(x_train))
    np.save(config.X_VAL_ZSCORE_NPY, per_beat_zscore(x_val))
    np.save(config.X_TEST_ZSCORE_NPY, per_beat_zscore(x_test))

    np.save(config.TRAIN_INDICES_NPY, train_idx)
    np.save(config.VAL_INDICES_NPY, val_idx)

    summary = pd.DataFrame(
        [
            describe_waveforms("train", x_train),
            describe_waveforms("validation", x_val),
            describe_waveforms("test", x_test),
        ]
    )
    summary.to_csv(config.TABLES_DIR / "processed_split_summary.csv", index=False)

    split_counts = pd.DataFrame(
        [
            {"split": "train", **class_distribution(y_train)},
            {"split": "validation", **class_distribution(y_val)},
            {"split": "test", **class_distribution(y_test)},
        ]
    ).fillna(0)
    split_counts.to_csv(config.TABLES_DIR / "processed_split_class_counts.csv", index=False)

    overlap = np.intersect1d(train_idx, val_idx)
    if len(overlap) != 0:
        raise RuntimeError("Train and validation indices overlap.")

    print("Waveform summary")
    print(summary.to_string(index=False))
    print("\nSplit class counts")
    print(split_counts.to_string(index=False))
    print(f"\nTrain/validation index overlap: {len(overlap)}")


if __name__ == "__main__":
    main()
