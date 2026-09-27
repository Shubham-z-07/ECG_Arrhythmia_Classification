import hashlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.models.evaluate import metric_summary


def row_hashes(x: np.ndarray) -> set[str]:
    rounded = np.round(x.astype(np.float32), decimals=6)
    return {hashlib.sha1(row.tobytes()).hexdigest() for row in rounded}


def split_overlap_checks() -> pd.DataFrame:
    train_idx = np.load(config.TRAIN_INDICES_NPY)
    val_idx = np.load(config.VAL_INDICES_NPY)
    x_train = np.load(config.X_TRAIN_RAW_NPY)
    x_val = np.load(config.X_VAL_RAW_NPY)
    x_test = np.load(config.X_TEST_RAW_NPY)

    train_hashes = row_hashes(x_train)
    val_hashes = row_hashes(x_val)
    test_hashes = row_hashes(x_test)

    rows = [
        {
            "check": "train_validation_index_overlap",
            "value": int(np.intersect1d(train_idx, val_idx).size),
            "notes": "Must be zero.",
        },
        {
            "check": "train_validation_duplicate_waveforms",
            "value": len(train_hashes & val_hashes),
            "notes": "Exact rounded waveform duplicates across the derived validation split.",
        },
        {
            "check": "train_test_duplicate_waveforms",
            "value": len(train_hashes & test_hashes),
            "notes": "Exact rounded waveform duplicates across Kaggle train and test files.",
        },
        {
            "check": "validation_test_duplicate_waveforms",
            "value": len(val_hashes & test_hashes),
            "notes": "Exact rounded waveform duplicates across validation and Kaggle test.",
        },
    ]
    return pd.DataFrame(rows)


def shuffled_label_models() -> dict[str, object]:
    return {
        "shuffled_logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=config.RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "shuffled_random_forest": RandomForestClassifier(
            n_estimators=80,
            max_depth=14,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=config.RANDOM_SEED,
        ),
    }


def shuffled_label_check(max_train_rows: int = 25000) -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_SEED)
    x_train = np.load(config.FEATURES_TRAIN_NPY)
    y_train = np.load(config.Y_TRAIN_NPY)
    x_val = np.load(config.FEATURES_VAL_NPY)
    y_val = np.load(config.Y_VAL_NPY)

    if len(y_train) > max_train_rows:
        sample_idx = rng.choice(len(y_train), size=max_train_rows, replace=False)
        x_train = x_train[sample_idx]
        y_train = y_train[sample_idx]

    shuffled_y = rng.permutation(y_train)
    rows = []
    for name, model in shuffled_label_models().items():
        model.fit(x_train, shuffled_y)
        pred_val = model.predict(x_val)
        rows.append(metric_summary(y_val, pred_val, name, "validation"))
    return pd.DataFrame(rows)


def main() -> None:
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    split_checks = split_overlap_checks()
    shuffled = shuffled_label_check()

    split_checks.to_csv(config.SANITY_SPLIT_CHECKS_CSV, index=False)
    shuffled.to_csv(config.SANITY_SHUFFLED_LABELS_CSV, index=False)

    print("Split checks")
    print(split_checks.to_string(index=False))
    print("\nShuffled-label validation metrics")
    print(shuffled.to_string(index=False))


if __name__ == "__main__":
    main()
