"""Signal preprocessing utilities used by the ECG ML workflow."""

from src.features.signal_processing import butterworth_filter


def preprocess_signal(x):
    return butterworth_filter(x)


__all__ = ["preprocess_signal", "butterworth_filter"]
