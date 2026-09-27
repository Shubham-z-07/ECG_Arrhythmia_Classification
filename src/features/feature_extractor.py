"""Feature extraction entry point for the ECG ML pipeline."""

from src.features.extract_features import extract_features, save_feature_outputs


def extract_heartbeat_features(x, apply_filter=True):
    return extract_features(x, apply_filter=apply_filter)


__all__ = ["extract_features", "save_feature_outputs", "extract_heartbeat_features"]
