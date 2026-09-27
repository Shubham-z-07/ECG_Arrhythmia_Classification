from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
METRICS_DIR = MODELS_DIR / "metrics"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

MITBIH_TRAIN_CSV = RAW_DATA_DIR / "mitbih_train.csv"
MITBIH_TEST_CSV = RAW_DATA_DIR / "mitbih_test.csv"

RANDOM_SEED = 42
VALIDATION_SIZE = 0.20
LABEL_COLUMN_INDEX = -1
EXPECTED_NUM_CLASSES = 5

X_TRAIN_RAW_NPY = PROCESSED_DATA_DIR / "x_train_raw.npy"
Y_TRAIN_NPY = PROCESSED_DATA_DIR / "y_train.npy"
X_VAL_RAW_NPY = PROCESSED_DATA_DIR / "x_val_raw.npy"
Y_VAL_NPY = PROCESSED_DATA_DIR / "y_val.npy"
X_TEST_RAW_NPY = PROCESSED_DATA_DIR / "x_test_raw.npy"
Y_TEST_NPY = PROCESSED_DATA_DIR / "y_test.npy"

X_TRAIN_ZSCORE_NPY = PROCESSED_DATA_DIR / "x_train_zscore.npy"
X_VAL_ZSCORE_NPY = PROCESSED_DATA_DIR / "x_val_zscore.npy"
X_TEST_ZSCORE_NPY = PROCESSED_DATA_DIR / "x_test_zscore.npy"

TRAIN_INDICES_NPY = PROCESSED_DATA_DIR / "train_indices.npy"
VAL_INDICES_NPY = PROCESSED_DATA_DIR / "val_indices.npy"

FEATURES_TRAIN_NPY = PROCESSED_DATA_DIR / "features_train.npy"
FEATURES_VAL_NPY = PROCESSED_DATA_DIR / "features_val.npy"
FEATURES_TEST_NPY = PROCESSED_DATA_DIR / "features_test.npy"
FEATURE_NAMES_TXT = PROCESSED_DATA_DIR / "feature_names.txt"

CLASSICAL_MODEL_SUMMARY_CSV = METRICS_DIR / "classical_model_summary.csv"
CLASSICAL_PER_CLASS_CSV = METRICS_DIR / "classical_per_class_metrics.csv"
CLASSICAL_TUNING_CSV = METRICS_DIR / "classical_validation_tuning.csv"

SANITY_SPLIT_CHECKS_CSV = METRICS_DIR / "sanity_split_checks.csv"
SANITY_SHUFFLED_LABELS_CSV = METRICS_DIR / "sanity_shuffled_labels.csv"
