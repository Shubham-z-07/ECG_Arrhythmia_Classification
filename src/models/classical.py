import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from src import config
from src.models.evaluate import confusion_matrix_df, metric_summary, per_class_metrics
from src.visualization.result_plots import save_confusion_matrix, save_metric_barplot

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: BaseEstimator
    param_grid: list[dict]
    uses_sample_weight: bool = False


def load_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.load(config.FEATURES_TRAIN_NPY),
        np.load(config.Y_TRAIN_NPY),
        np.load(config.FEATURES_VAL_NPY),
        np.load(config.Y_VAL_NPY),
        np.load(config.FEATURES_TEST_NPY),
        np.load(config.Y_TEST_NPY),
    )


def model_specs() -> list[ModelSpec]:
    specs = [
        ModelSpec(
            name="logistic_regression",
            estimator=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=3000,
                            random_state=config.RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            param_grid=[
                {"model__C": [0.1]},
                {"model__C": [1.0]},
                {"model__C": [3.0]},
            ],
        ),
        ModelSpec(
            name="linear_svm",
            estimator=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LinearSVC(
                            class_weight="balanced",
                            max_iter=8000,
                            random_state=config.RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            param_grid=[
                {"model__C": [0.1]},
                {"model__C": [1.0]},
            ],
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                n_estimators=250,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=config.RANDOM_SEED,
            ),
            param_grid=[
                {"max_depth": [12], "min_samples_leaf": [1]},
                {"max_depth": [18], "min_samples_leaf": [1]},
                {"max_depth": [None], "min_samples_leaf": [1]},
            ],
        ),
    ]

    if XGBClassifier is not None:
        specs.append(
            ModelSpec(
                name="xgboost",
                estimator=XGBClassifier(
                    objective="multi:softprob",
                    num_class=config.EXPECTED_NUM_CLASSES,
                    eval_metric="mlogloss",
                    n_estimators=250,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=config.RANDOM_SEED,
                ),
                param_grid=[
                    {"max_depth": [3], "learning_rate": [0.08]},
                    {"max_depth": [5], "learning_rate": [0.06]},
                ],
                uses_sample_weight=True,
            )
        )

    return specs


def fit_with_optional_weights(
    estimator: BaseEstimator,
    x: np.ndarray,
    y: np.ndarray,
    uses_sample_weight: bool,
) -> BaseEstimator:
    if uses_sample_weight:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        estimator.fit(x, y, sample_weight=sample_weight)
    else:
        estimator.fit(x, y)
    return estimator


def select_model(
    spec: ModelSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[BaseEstimator, dict, list[dict]]:
    best_model = None
    best_params = None
    best_macro_f1 = -np.inf
    tuning_rows = []

    for params in ParameterGrid(spec.param_grid):
        estimator = clone(spec.estimator).set_params(**params)
        estimator = fit_with_optional_weights(estimator, x_train, y_train, spec.uses_sample_weight)
        pred_val = estimator.predict(x_val)
        scores = metric_summary(y_val, pred_val, spec.name, "validation")
        scores["params"] = json.dumps(params, sort_keys=True)
        tuning_rows.append(scores)

        if scores["macro_f1"] > best_macro_f1:
            best_macro_f1 = scores["macro_f1"]
            best_params = params
            best_model = estimator

    if best_model is None or best_params is None:
        raise RuntimeError(f"No model was selected for {spec.name}.")
    return best_model, best_params, tuning_rows


def evaluate_and_save(
    model: BaseEstimator,
    model_name: str,
    best_params: dict,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[list[dict], list[pd.DataFrame]]:
    labels = list(range(config.EXPECTED_NUM_CLASSES))
    metric_rows = []
    per_class_rows = []

    for split, x, y in [("validation", x_val, y_val), ("test", x_test, y_test)]:
        pred = model.predict(x)
        summary = metric_summary(y, pred, model_name, split)
        summary["params"] = json.dumps(best_params, sort_keys=True)
        metric_rows.append(summary)
        per_class_rows.append(per_class_metrics(y, pred, model_name, split))

        matrix = confusion_matrix_df(y, pred, labels=labels)
        matrix_path = config.METRICS_DIR / f"{model_name}_{split}_confusion_matrix.csv"
        matrix.to_csv(matrix_path)
        save_confusion_matrix(
            matrix,
            config.FIGURES_DIR / f"{model_name}_{split}_confusion_matrix.png",
            title=f"{model_name} {split} confusion matrix",
        )

    return metric_rows, per_class_rows


def main() -> None:
    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val, x_test, y_test = load_features()

    all_metric_rows = []
    all_per_class = []
    all_tuning_rows = []

    for spec in model_specs():
        model, best_params, tuning_rows = select_model(spec, x_train, y_train, x_val, y_val)
        all_tuning_rows.extend(tuning_rows)
        metric_rows, per_class_rows = evaluate_and_save(
            model,
            spec.name,
            best_params,
            x_val,
            y_val,
            x_test,
            y_test,
        )
        all_metric_rows.extend(metric_rows)
        all_per_class.extend(per_class_rows)
        joblib.dump(model, config.CHECKPOINTS_DIR / f"{spec.name}.joblib")

    summary = pd.DataFrame(all_metric_rows).sort_values(["split", "macro_f1"], ascending=[True, False])
    per_class = pd.concat(all_per_class, ignore_index=True)
    tuning = pd.DataFrame(all_tuning_rows).sort_values(["model", "macro_f1"], ascending=[True, False])

    summary.to_csv(config.CLASSICAL_MODEL_SUMMARY_CSV, index=False)
    per_class.to_csv(config.CLASSICAL_PER_CLASS_CSV, index=False)
    tuning.to_csv(config.CLASSICAL_TUNING_CSV, index=False)
    save_metric_barplot(summary, config.FIGURES_DIR / "classical_model_metrics.png")

    print("ML model summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
