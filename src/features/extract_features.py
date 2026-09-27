import numpy as np
import pandas as pd
from scipy import fft, signal, stats

from src import config
from src.features.signal_processing import butterworth_filter, dominant_peak_features


def zero_crossing_rate(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=1, keepdims=True)
    return np.mean(np.diff(np.signbit(centered), axis=1), axis=1)


def time_domain_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    diff = np.diff(x, axis=1)
    q25, q50, q75 = np.quantile(x, [0.25, 0.50, 0.75], axis=1)

    values = np.column_stack(
        [
            x.mean(axis=1),
            x.std(axis=1),
            x.min(axis=1),
            x.max(axis=1),
            np.ptp(x, axis=1),
            q25,
            q50,
            q75,
            stats.skew(x, axis=1, bias=False),
            stats.kurtosis(x, axis=1, bias=False),
            np.sum(x**2, axis=1),
            np.mean(np.abs(diff), axis=1),
            np.max(np.abs(diff), axis=1),
            zero_crossing_rate(x),
            np.mean(x == 0.0, axis=1),
        ]
    )

    names = [
        "mean",
        "std",
        "min",
        "max",
        "range",
        "q25",
        "median",
        "q75",
        "skew",
        "kurtosis",
        "energy",
        "mean_abs_diff",
        "max_abs_diff",
        "zero_crossing_rate",
        "zero_fraction",
    ]
    return values.astype(np.float32), names


def frequency_domain_features(x: np.ndarray, fs: float = 125.0) -> tuple[np.ndarray, list[str]]:
    spectrum = np.abs(fft.rfft(x, axis=1)) ** 2
    freqs = fft.rfftfreq(x.shape[1], d=1.0 / fs)
    total_power = np.maximum(spectrum.sum(axis=1), 1e-12)

    bands = [(0.0, 5.0), (5.0, 15.0), (15.0, 30.0), (30.0, 62.5)]
    band_values = []
    band_names = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        band_power = spectrum[:, mask].sum(axis=1)
        band_values.append(band_power / total_power)
        band_names.append(f"fft_power_{low:g}_{high:g}hz")

    centroid = (spectrum * freqs).sum(axis=1) / total_power
    cumulative_power = np.cumsum(spectrum, axis=1) / total_power[:, None]
    rolloff_idx = np.argmax(cumulative_power >= 0.85, axis=1)
    rolloff = freqs[rolloff_idx]
    dominant_freq = freqs[np.argmax(spectrum[:, 1:], axis=1) + 1]
    spectral_entropy = stats.entropy(spectrum / total_power[:, None], axis=1)

    values = np.column_stack(
        band_values + [centroid, rolloff, dominant_freq, spectral_entropy]
    )
    names = band_names + [
        "spectral_centroid",
        "spectral_rolloff_85",
        "dominant_frequency",
        "spectral_entropy",
    ]
    return values.astype(np.float32), names


def shape_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    peaks = dominant_peak_features(x)
    autocorr_lag1 = np.mean(x[:, 1:] * x[:, :-1], axis=1)
    autocorr_lag5 = np.mean(x[:, 5:] * x[:, :-5], axis=1)

    values = np.column_stack([peaks, autocorr_lag1, autocorr_lag5])
    names = [
        "main_peak_position",
        "main_peak_height",
        "main_peak_prominence",
        "main_peak_width",
        "autocorr_lag1",
        "autocorr_lag5",
    ]
    return values.astype(np.float32), names


def extract_features(x: np.ndarray, apply_filter: bool = True) -> tuple[np.ndarray, list[str]]:
    filtered = butterworth_filter(x) if apply_filter else x.astype(np.float32, copy=False)

    blocks = [
        time_domain_features(filtered),
        frequency_domain_features(filtered),
        shape_features(filtered),
    ]
    values = np.column_stack([block[0] for block in blocks]).astype(np.float32)
    names = [name for block in blocks for name in block[1]]
    return values, names


def load_processed_waveforms() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.load(config.X_TRAIN_RAW_NPY),
        np.load(config.X_VAL_RAW_NPY),
        np.load(config.X_TEST_RAW_NPY),
    )


def save_feature_outputs() -> None:
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    x_train, x_val, x_test = load_processed_waveforms()
    features_train, names = extract_features(x_train)
    features_val, names_val = extract_features(x_val)
    features_test, names_test = extract_features(x_test)

    if names != names_val or names != names_test:
        raise RuntimeError("Feature names differ across splits.")

    np.save(config.FEATURES_TRAIN_NPY, features_train)
    np.save(config.FEATURES_VAL_NPY, features_val)
    np.save(config.FEATURES_TEST_NPY, features_test)
    config.FEATURE_NAMES_TXT.write_text("\n".join(names) + "\n", encoding="utf-8")

    summary = pd.DataFrame(
        [
            {"split": "train", "rows": features_train.shape[0], "features": features_train.shape[1]},
            {"split": "validation", "rows": features_val.shape[0], "features": features_val.shape[1]},
            {"split": "test", "rows": features_test.shape[0], "features": features_test.shape[1]},
        ]
    )
    summary.to_csv(config.TABLES_DIR / "feature_matrix_summary.csv", index=False)

    print("Feature matrix summary")
    print(summary.to_string(index=False))
    print(f"\nFeature count: {len(names)}")


if __name__ == "__main__":
    save_feature_outputs()
