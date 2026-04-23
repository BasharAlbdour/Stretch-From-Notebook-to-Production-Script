# Petra Telecom Churn Model Comparison — Production CLI

A production-quality command-line tool that compares 6 churn prediction models
using 5-fold stratified cross-validation and produces a full evaluation report.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python compare_models.py --data-path  [options]
```

## Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data-path` | Yes | — | Path to input dataset CSV |
| `--output-dir` | No | `./output` | Directory to save all results |
| `--n-folds` | No | `5` | Number of CV folds |
| `--random-seed` | No | `42` | Random seed for reproducibility |
| `--dry-run` | No | `False` | Validate data and print config without training |

## Example Commands

**Normal run:**
```bash
python compare_models.py --data-path data/telecom_churn.csv
```

**Dry run — validate data and print config without training:**
```bash
python compare_models.py --data-path data/telecom_churn.csv --dry-run
```

**Custom output directory and folds:**
```bash
python compare_models.py --data-path data/telecom_churn.csv --output-dir results/ --n-folds 10
```

**Custom random seed:**
```bash
python compare_models.py --data-path data/telecom_churn.csv --random-seed 123
```

## Output Files

All files are saved to `--output-dir` (default: `./output`):

| File | Description |
|---|---|
| `comparison_table.csv` | CV metrics for all 6 models (mean ± std) |
| `experiment_log.csv` | Timestamped experiment log |
| `pr_curves.png` | Precision-recall curves for top 3 models |
| `calibration.png` | Calibration curves for top 3 models |
| `best_model.joblib` | Best model Pipeline saved with joblib |
| `threshold_sweep.png` | Threshold sweep plot for RF_default |
| `tree_vs_linear_disagreement.md` | Tree vs linear capability analysis |

## Models Compared

| Model | Description |
|---|---|
| Dummy | Baseline — always predicts majority class |
| LR_default | Logistic Regression with StandardScaler |
| LR_balanced | Logistic Regression with class_weight='balanced' |
| DT_depth5 | Decision Tree (max_depth=5) |
| RF_default | Random Forest (100 trees, max_depth=10) |
| RF_balanced | Random Forest with class_weight='balanced' |

## Dataset Requirements

The input CSV must contain these columns:

**Features:** `tenure`, `monthly_charges`, `total_charges`, `num_support_calls`,
`senior_citizen`, `has_partner`, `has_dependents`, `contract_months`

**Target:** `churned` (binary: 0 or 1)

## How It Works

1. Loads and validates the dataset — exits with an error if columns are missing
2. Splits 80/20 with stratification to preserve the churn rate in both sets
3. Runs stratified cross-validation across all 6 model configurations
4. Fits all models on the full training set for plot generation
5. Saves PR curves, calibration curves, and threshold sweep plots
6. Persists the best model (by PR-AUC) as a joblib artifact
7. Finds the sample where RF and LR disagree most and explains why structurally

## Loading the Best Model

```python
from joblib import load

model = load("output/best_model.joblib")
predictions  = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]
```

## Design Notes

- Built with `argparse` — run `python compare_models.py --help` for full usage
- All progress is logged with Python's `logging` module at appropriate levels:
  - `INFO` — normal progress
  - `WARNING` — potential issues (e.g. very low positive rate)
  - `ERROR` — fatal issues (missing file, missing columns) — exits with code 1
- `--dry-run` validates data and prints full pipeline config without fitting any models
- Script is importable as a module — all functions can be tested independently
- Refactored from `model_comparison.py` (Integration 5B base assignment)

## Repository Structure
```
Stretch-From-Notebook-to-Production-Script/
├── compare_models.py       ← Production CLI script (stretch deliverable)
├── model_comparison.py     ← Original Integration 5B script
├── README.md               ← This file
├── requirements.txt        ← Python dependencies
├── data/
│   └── telecom_churn.csv   ← Petra Telecom dataset
└── output/                 ← Sample output from running compare_models.py
    ├── comparison_table.csv
    ├── experiment_log.csv
    ├── pr_curves.png
    ├── calibration.png
    ├── best_model.joblib
    ├── threshold_sweep.png
    └── tree_vs_linear_disagreement.md
    └── sample_output.txt
```

## Why model_comparison.py is included

The stretch assignment asks to "refactor the pipeline from Integration 5B into a 
production CLI tool." `model_comparison.py` is the original Integration 5B script 
that `compare_models.py` was refactored from. Including both files makes the 
before/after comparison explicit:

- `model_comparison.py` — notebook-style script with hardcoded paths, print 
  statements, and no CLI interface
- `compare_models.py` — production CLI with argparse, structured logging, 
  --dry-run mode, and configurable arguments

The key differences between the two scripts:

| Feature | model_comparison.py | compare_models.py |
|---|---|---|
| Arguments | Hardcoded | CLI via argparse |
| Output | print() | logging module |
| Data path | Fixed default | --data-path required |
| Validation | None | --dry-run mode |
| Output dir | Hardcoded results/ | --output-dir configurable |
| Error handling | None | sys.exit(1) with [ERROR] log |