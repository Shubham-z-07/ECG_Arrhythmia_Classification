"""Filtering utilities for ECG segmentation."""

from src.features.signal_processing import butterworth_filter


def filtered_signal(x):
    return butterworth_filter(x)


__all__ = ["filtered_signal", "butterworth_filter"]
