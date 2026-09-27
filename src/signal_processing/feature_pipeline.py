"""Compatibility wrapper around ECG feature extraction."""

from src.features.extract_features import (
    extract_features,
    save_feature_outputs,
)


def main() -> None:
    save_feature_outputs()


if __name__ == "__main__":
    main()
