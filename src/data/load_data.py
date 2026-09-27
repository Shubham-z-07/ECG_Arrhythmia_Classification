from pathlib import Path

import numpy as np
import pandas as pd

from src import config


def read_heartbeat_csv(path: Path) -> pd.DataFrame:
    """Read a Kaggle heartbeat CSV with no header row."""
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return pd.read_csv(path, header=None, dtype=np.float32)


def split_features_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split waveform columns from the final label column."""
    x = df.iloc[:, : config.LABEL_COLUMN_INDEX].to_numpy(dtype=np.float32)
    y = df.iloc[:, config.LABEL_COLUMN_INDEX].to_numpy(dtype=np.int64)
    return x, y


def load_mitbih() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_df = read_heartbeat_csv(config.MITBIH_TRAIN_CSV)
    test_df = read_heartbeat_csv(config.MITBIH_TEST_CSV)
    x_train, y_train = split_features_labels(train_df)
    x_test, y_test = split_features_labels(test_df)
    return x_train, y_train, x_test, y_test
