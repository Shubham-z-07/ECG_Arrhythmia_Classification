"""Final reporting entry point for the ML ECG pipeline."""

from src.models.final_report import main as generate_ml_report


def main() -> None:
    generate_ml_report()


if __name__ == "__main__":
    main()
