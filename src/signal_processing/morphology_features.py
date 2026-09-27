"""Morphology feature extraction utilities for ECG beats."""

from src.features.signal_processing import dominant_peak_features


def extract_peak_features(x):
    return dominant_peak_features(x)


__all__ = ["extract_peak_features", "dominant_peak_features"]
