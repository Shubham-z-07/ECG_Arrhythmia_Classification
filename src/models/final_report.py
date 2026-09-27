import pandas as pd

from src import config


def main() -> None:
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    classical = pd.read_csv(config.CLASSICAL_MODEL_SUMMARY_CSV)
    comparison = classical[classical["split"] == "test"].copy()
    comparison = comparison[["model", "split", "accuracy", "macro_f1", "weighted_f1"]]
    comparison = comparison.sort_values("macro_f1", ascending=False)
    comparison.to_csv(config.TABLES_DIR / "final_model_comparison.csv", index=False)

    per_class = pd.read_csv(config.CLASSICAL_PER_CLASS_CSV)
    per_class = per_class[
        (per_class["split"] == "test") & (per_class["model"] == "random_forest")
    ].sort_values("class")
    per_class.to_csv(config.TABLES_DIR / "final_per_class_random_forest.csv", index=False)

    print("Final ML model comparison")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
