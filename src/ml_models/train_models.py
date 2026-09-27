"""ML-oriented training entry point for ECG heartbeat classification."""

from src.models.classical import main as run_ml_training


def main() -> None:
    run_ml_training()


if __name__ == "__main__":
    main()
