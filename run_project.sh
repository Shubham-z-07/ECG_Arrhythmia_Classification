#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
    echo "Error: virtual environment not found at $PROJECT_ROOT/.venv" >&2
    echo "Create it with: python3 -m venv .venv" >&2
    exit 1
fi

source "$PROJECT_ROOT/.venv/bin/activate"

python -m src.data.validate_data

python -m src.data.split_data

python -m src.signal_processing.feature_pipeline

python -m src.ml_models.train_models

python -m src.models.sanity_checks

python -m src.ml_models.final_report
