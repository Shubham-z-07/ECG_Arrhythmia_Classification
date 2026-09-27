"""Frequency-domain ECG feature helpers."""

from src.features.extract_features import frequency_domain_features


def extract_fft_features(x, fs=125.0):
    return frequency_domain_features(x, fs=fs)


__all__ = ["frequency_domain_features", "extract_fft_features"]
