# ECG_Arrhythmia_Classification
ECG Classification Using Signal Processing [Machine learning]

This project is a machine learning ECG heartbeat classification system built around signal processing and engineered waveform features. The goal is to detect whether a heartbeat is normal or arrhythmic and, when abnormal, classify it into one of the five MIT-BIH rhythm categories: N, S, V, F, and Q.

The project uses signal-processing-driven feature extraction combined with ML models, not deep learning. The workflow is designed to be interpretable, robust to class imbalance, and suitable for a serious ECG classification portfolio project.

## Project goal

This is a multiclass ECG classification task with a clinically meaningful framing:

- detect normal vs abnormal rhythm
- identify the specific arrhythmia category
- handle class imbalance using macro F1 and per-class evaluation
- evaluate the strongest ML model on a hold-out test set

## Dataset

Dataset: https://www.kaggle.com/datasets/shayanfazeli/heartbeat

This project uses the MIT-BIH heartbeat files:

- data/raw/mitbih_train.csv
- data/raw/mitbih_test.csv

The PTBDB files are excluded because they represent a different classification task and would dilute the MIT-BIH 5-class ECG problem.

The MIT-BIH files contain 188 columns per row: 187 ECG signal samples and one label column.

| Class | Label | Train | Validation | Test |
|---:|---|---:|---:|---:|
| 0 | N | 57,977 | 14,494 | 18,118 |
| 1 | S | 1,778 | 445 | 556 |
| 2 | V | 4,630 | 1,158 | 1,448 |
| 3 | F | 513 | 128 | 162 |
| 4 | Q | 5,145 | 1,286 | 1,608 |

## Method

### Pipeline overview

1. Validate the raw ECG dataset
2. Split the MIT-BIH training set into train and validation stratified folds
3. Apply signal preprocessing and filtering
4. Extract waveform and spectral features
5. Train ML models using engineered features
6. Select the best model using validation macro F1
7. Evaluate the final model on the hold-out test set
8. Generate confusion matrices and per-class performance summaries

### Signal-processing feature pipeline

The feature extraction stage combines signal morphology, temporal behavior, and frequency-domain information:

- time-domain statistics: mean, std, min, max, range, quantiles, skew, kurtosis, energy, zero-crossing rate
- frequency-domain descriptors: FFT band power, spectral centroid, spectral rolloff, dominant frequency, spectral entropy
- waveform-shape descriptors: main peak position, peak height, prominence, width, autocorrelation patterns

These features are saved as:

- data/processed/features_train.npy
- data/processed/features_val.npy
- data/processed/features_test.npy

### ML models

The project evaluates and selects among:

- Logistic Regression
- Linear SVM
- Random Forest
- XGBoost

The selected final model is chosen using validation macro F1, which is the right metric for this strongly imbalanced ECG dataset.

## Verified benchmark result

The current ML benchmark shows the best model is Random Forest.

| Model | Test accuracy | Test macro F1 | Test weighted F1 |
|---|---:|---:|---:|
| Random Forest | 0.9712 | 0.8584 | 0.9691 |
| XGBoost | 0.9380 | 0.7902 | 0.9437 |
| Linear SVM | 0.9036 | 0.6400 | 0.9087 |
| Logistic Regression | 0.7136 | 0.5266 | 0.7801 |

This confirms that the signal-processing + ML approach is the right foundation for the project.

## Main finding

Random Forest is the most reliable model in this setup because it captures the nonlinear beat morphology patterns while preserving strong minority-class performance. Macro F1 remains the central evaluation metric because the dataset is highly imbalanced and class 0 is dominant.

## Sanity checks

The project verifies data integrity and split quality:

- train/validation index overlap: 0
- train/validation duplicate waveforms: 0
- train/test duplicate waveforms: 0
- validation/test duplicate waveforms: 0

The shuffled-label validation also confirms that the model is learning real heartbeat structure instead of only exploiting class imbalance.

## Reproducing the pipeline

Use Python 3.14.7, the latest stable release, or Python 3.12 or newer. The
dependency versions in `requirements.txt` are the latest releases available
when this project was updated and include only packages used by the source code
or notebooks.

Create the virtual environment from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the Kaggle CSV files in the raw directory:

- data/raw/mitbih_train.csv
- data/raw/mitbih_test.csv

Then run:

```bash
python -m src.data.validate_data
python -m src.data.split_data
python -m src.signal_processing.feature_pipeline
python -m src.ml_models.train_models
python -m src.models.sanity_checks
python -m src.ml_models.final_report
```

## Project structure

```text
ecg-classification-signal-processing-ml/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
│   ├── checkpoints/
│   └── metrics/
├── notebooks/
├── reports/
│   ├── figures/
│   └── tables/
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── load_data.py
│   │   ├── split_data.py
│   │   └── validate_data.py
│   ├── features/
│   │   ├── extract_features.py
│   │   └── signal_processing.py
│   ├── models/
│   │   ├── classical.py
│   │   ├── evaluate.py
│   │   ├── final_report.py
│   │   └── sanity_checks.py
│   ├── evaluation/
│   │   └── sanity_checks.py
│   ├── reporting/
│   │   └── final_report.py
│   ├── pipeline/
│   │   └── run_training.py
│   └── visualization/
│       ├── eda_plots.py
│       └── result_plots.py
├── requirements.txt
├── README.md
└── LICENSE
```

## Advanced improvement path

This project is already a strong ML foundation. The next advanced refinements are focused and appropriate:

- patient-aware splitting for more reliable generalization
- Fourier and spectral feature expansion
- better class-balancing and threshold tuning
- stronger per-class error analysis
- deployment-ready evaluation workflow

The important point is that the project remains ML-first, interpretable, and grounded in signal processing rather than deep learning.
