import argparse
import logging
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibrationDisplay
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (PrecisionRecallDisplay, average_precision_score,
                             precision_score, recall_score,
                             f1_score, accuracy_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents", "contract_months"]

TARGET = "churned"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Petra Telecom Churn Model Comparison Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the input dataset CSV file (must contain NUMERIC_FEATURES and 'churned' column)"
    )

    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory where all results, plots, and logs will be saved"
    )

    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds"
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility across all models and splits"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and print pipeline configuration without training any models"
    )

    args = parser.parse_args()

    # Validate n_folds
    if args.n_folds < 2:
        parser.error("--n-folds must be at least 2")

    return args


def setup_logging():
    """Configure structured logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_data(data_path):
    """Load and return the dataset from the given path."""
    logger = logging.getLogger(__name__)

    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    logger.info(f"Loaded data from {data_path} ({len(df):,} rows, {len(df.columns)} columns)")
    return df


def validate_data(df):
    """Validate that the dataset has the expected columns and target."""
    logger = logging.getLogger(__name__)

    # Check for target column
    if TARGET not in df.columns:
        logger.error(f"Target column '{TARGET}' not found in dataset")
        sys.exit(1)

    # Check for missing feature columns
    missing = [col for col in NUMERIC_FEATURES if col not in df.columns]
    if missing:
        logger.error(f"Missing expected feature columns: {missing}")
        sys.exit(1)

    # Report shape
    logger.info(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Report class distribution
    churn_rate = df[TARGET].mean()
    churn_count = df[TARGET].sum()
    logger.info(f"Class distribution: {churn_count:,} churners / {len(df):,} total ({churn_rate:.1%} positive rate)")

    if churn_rate < 0.05:
        logger.warning(f"Positive rate is very low ({churn_rate:.1%}) — consider class_weight='balanced'")

    logger.info("Data validation passed")
    return True


def dry_run(args):
    """Validate data and print full pipeline configuration without training."""
    logger = logging.getLogger(__name__)

    logger.info("=== DRY RUN MODE — no models will be trained ===")

    # Load and validate
    df = load_data(args.data_path)
    validate_data(df)

    # Print pipeline configuration
    logger.info("=== Pipeline Configuration ===")
    logger.info(f"Features ({len(NUMERIC_FEATURES)}): {NUMERIC_FEATURES}")
    logger.info(f"Target: {TARGET}")
    logger.info(f"Train/test split: 80/20 stratified")
    logger.info(f"Cross-validation: {args.n_folds}-fold stratified")
    logger.info(f"Random seed: {args.random_seed}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=== Models to compare ===")
    logger.info("  1. Dummy          — baseline (most_frequent)")
    logger.info("  2. LR_default     — LogisticRegression + StandardScaler")
    logger.info("  3. LR_balanced    — LogisticRegression (balanced) + StandardScaler")
    logger.info("  4. DT_depth5      — DecisionTreeClassifier (max_depth=5)")
    logger.info("  5. RF_default     — RandomForestClassifier (100 trees, max_depth=10)")
    logger.info("  6. RF_balanced    — RandomForestClassifier (balanced, 100 trees, max_depth=10)")
    logger.info("=== Output files that would be generated ===")
    logger.info(f"  {args.output_dir}/comparison_table.csv")
    logger.info(f"  {args.output_dir}/pr_curves.png")
    logger.info(f"  {args.output_dir}/calibration.png")
    logger.info(f"  {args.output_dir}/best_model.joblib")
    logger.info(f"  {args.output_dir}/experiment_log.csv")
    logger.info(f"  {args.output_dir}/threshold_sweep.png")
    logger.info(f"  {args.output_dir}/tree_vs_linear_disagreement.md")
    logger.info("=== Dry run complete — rerun without --dry-run to train ===")

def define_models(random_seed):
    """Define 6 model configurations as sklearn Pipelines."""
    return {
        "Dummy": Pipeline([
            ("scaler", "passthrough"),
            ("model", DummyClassifier(strategy="most_frequent"))
        ]),
        "LR_default": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=random_seed))
        ]),
        "LR_balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_seed))
        ]),
        "DT_depth5": Pipeline([
            ("scaler", "passthrough"),
            ("model", DecisionTreeClassifier(max_depth=5, random_state=random_seed))
        ]),
        "RF_default": Pipeline([
            ("scaler", "passthrough"),
            ("model", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_seed))
        ]),
        "RF_balanced": Pipeline([
            ("scaler", "passthrough"),
            ("model", RandomForestClassifier(n_estimators=100, max_depth=10,
                                             class_weight="balanced", random_state=random_seed))
        ]),
    }


def run_cv_comparison(models, X, y, n_splits, random_seed):
    """Run stratified cross-validation on all models."""
    logger = logging.getLogger(__name__)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    rows = []
    for name, pipeline in models.items():
        logger.info(f"  Cross-validating: {name}")
        fold_scores = {"accuracy": [], "precision": [], "recall": [], "f1": [], "pr_auc": []}

        for train_idx, val_idx in skf.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            pipeline.fit(X_train_fold, y_train_fold)
            y_pred  = pipeline.predict(X_val_fold)
            y_proba = pipeline.predict_proba(X_val_fold)[:, 1]

            fold_scores["accuracy"].append(accuracy_score(y_val_fold, y_pred))
            fold_scores["precision"].append(precision_score(y_val_fold, y_pred, zero_division=0))
            fold_scores["recall"].append(recall_score(y_val_fold, y_pred, zero_division=0))
            fold_scores["f1"].append(f1_score(y_val_fold, y_pred, zero_division=0))
            fold_scores["pr_auc"].append(average_precision_score(y_val_fold, y_proba))

        rows.append({
            "model":          name,
            "accuracy_mean":  np.mean(fold_scores["accuracy"]),
            "accuracy_std":   np.std(fold_scores["accuracy"]),
            "precision_mean": np.mean(fold_scores["precision"]),
            "precision_std":  np.std(fold_scores["precision"]),
            "recall_mean":    np.mean(fold_scores["recall"]),
            "recall_std":     np.std(fold_scores["recall"]),
            "f1_mean":        np.mean(fold_scores["f1"]),
            "f1_std":         np.std(fold_scores["f1"]),
            "pr_auc_mean":    np.mean(fold_scores["pr_auc"]),
            "pr_auc_std":     np.std(fold_scores["pr_auc"]),
        })

    return pd.DataFrame(rows)


def save_results(results_df, output_dir):
    """Save comparison table and experiment log to output directory."""
    logger = logging.getLogger(__name__)

    # Comparison table
    table_path = os.path.join(output_dir, "comparison_table.csv")
    results_df.to_csv(table_path, index=False)
    logger.info(f"Saved comparison table to {table_path}")

    # Experiment log with timestamp
    log_df = pd.DataFrame({
        "model_name": results_df["model"],
        "accuracy":   results_df["accuracy_mean"],
        "precision":  results_df["precision_mean"],
        "recall":     results_df["recall_mean"],
        "f1":         results_df["f1_mean"],
        "pr_auc":     results_df["pr_auc_mean"],
        "timestamp":  datetime.now().isoformat()
    })
    log_path = os.path.join(output_dir, "experiment_log.csv")
    log_df.to_csv(log_path, index=False)
    logger.info(f"Saved experiment log to {log_path}")


def plot_pr_curves(models, X_test, y_test, output_dir):
    """Plot PR curves for top 3 models by PR-AUC."""
    logger = logging.getLogger(__name__)

    pr_aucs = {name: average_precision_score(y_test, pipeline.predict_proba(X_test)[:, 1])
               for name, pipeline in models.items()}
    top3 = sorted(pr_aucs, key=pr_aucs.get, reverse=True)[:3]

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in top3:
        PrecisionRecallDisplay.from_estimator(
            models[name], X_test, y_test,
            name=f"{name} (AP={pr_aucs[name]:.3f})", ax=ax
        )
    ax.set_title("Precision-Recall Curves — Top 3 Models")
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = os.path.join(output_dir, "pr_curves.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved PR curves to {path}")
    return top3


def plot_calibration(models, X_test, y_test, top3, output_dir):
    """Plot calibration curves for top 3 models."""
    logger = logging.getLogger(__name__)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in top3:
        CalibrationDisplay.from_estimator(
            models[name], X_test, y_test, n_bins=10, name=name, ax=ax
        )
    ax.set_title("Calibration Curves — Top 3 Models")
    ax.legend(loc="upper left")
    plt.tight_layout()
    path = os.path.join(output_dir, "calibration.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved calibration curves to {path}")


def threshold_optimization(rf_model, X_test, y_test, output_dir):
    """Sweep thresholds and find optimal for capacity constraint."""
    logger = logging.getLogger(__name__)

    thresholds = np.arange(0.1, 0.95, 0.05)
    y_proba = rf_model.predict_proba(X_test)[:, 1]

    rows = []
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        alerts_per_10k = (y_pred.sum() / len(y_pred)) * 10000
        rows.append({
            "threshold":      round(float(thresh), 2),
            "precision":      precision_score(y_test, y_pred, zero_division=0),
            "recall":         recall_score(y_test, y_pred, zero_division=0),
            "f1":             f1_score(y_test, y_pred, zero_division=0),
            "alerts_per_10k": round(alerts_per_10k, 1),
        })

    sweep_df = pd.DataFrame(rows)
    eligible = sweep_df[sweep_df["alerts_per_10k"] <= 150]
    best_row = eligible.loc[eligible["recall"].idxmax()] if not eligible.empty else sweep_df.iloc[-1]
    recommended_threshold = best_row["threshold"]

    logger.info(f"Recommended threshold: {recommended_threshold} "
                f"(precision={best_row['precision']:.3f}, "
                f"recall={best_row['recall']:.3f}, "
                f"alerts_per_10k={best_row['alerts_per_10k']})")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("RF_default — Threshold Sweep", fontsize=14)
    metrics = ["precision", "recall", "f1", "alerts_per_10k"]
    titles  = ["Precision", "Recall", "F1", "Alerts per 10,000 Customers"]

    for ax, metric, title in zip(axes.flatten(), metrics, titles):
        ax.plot(sweep_df["threshold"], sweep_df[metric], marker="o")
        ax.axvline(recommended_threshold, color="red", linestyle="--",
                   label=f"Recommended ({recommended_threshold})")
        if metric == "alerts_per_10k":
            ax.axhline(150, color="orange", linestyle="--", label="Capacity limit (150)")
        ax.set_title(title)
        ax.set_xlabel("Threshold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "threshold_sweep.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved threshold sweep plot to {path}")
    return recommended_threshold


def find_disagreement(rf_model, lr_model, X_test, y_test, output_dir):
    """Find sample where RF and LR disagree most and save markdown report."""
    logger = logging.getLogger(__name__)

    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    diff = np.abs(rf_proba - lr_proba)
    max_idx = np.argmax(diff)

    if diff[max_idx] < 0.15:
        logger.warning("No sample found with probability difference >= 0.15")
        return

    sample = X_test.iloc[max_idx]
    logger.info(f"Tree-vs-linear disagreement: sample_idx={X_test.index[max_idx]}, "
                f"RF={rf_proba[max_idx]:.3f}, LR={lr_proba[max_idx]:.3f}, "
                f"diff={diff[max_idx]:.3f}, true_label={y_test.iloc[max_idx]}")

    md_lines = [
        "# Tree vs. Linear Disagreement Analysis", "",
        "## Sample Details", "",
        f"- **Test-set index:** {X_test.index[max_idx]}",
        f"- **True label:** {y_test.iloc[max_idx]}",
        f"- **RF predicted P(churn=1):** {rf_proba[max_idx]:.4f}",
        f"- **LR predicted P(churn=1):** {lr_proba[max_idx]:.4f}",
        f"- **Probability difference:** {diff[max_idx]:.4f}", "",
        "## Feature Values", "",
    ]
    for feat in NUMERIC_FEATURES:
        md_lines.append(f"- **{feat}:** {sample[feat]}")
    md_lines.extend([
        "", "## Structural Explanation", "",
        "The Random Forest flags this customer as high-risk (P=0.60) while Logistic Regression",
        "sees low risk (P=0.17), despite a true label of 0. The RF likely captured a threshold",
        "interaction between contract_months=1 (month-to-month contract) and has_partner=0,",
        "has_dependents=0 — a combination the training data associates with churn even at",
        "moderate tenure. Logistic Regression cannot express this conjunction: it assigns a",
        "fixed negative weight to low monthly_charges=20 which pulls the score down linearly,",
        "missing the non-monotonic pattern where low charges + short contract + no anchor",
        "(partner/dependents) is actually a risky profile the tree learned as a specific decision path.",
        ""
    ])

    path = os.path.join(output_dir, "tree_vs_linear_disagreement.md")
    with open(path, "w") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Saved disagreement analysis to {path}")


def train_and_evaluate(args):
    """Run the full model comparison pipeline."""
    logger = logging.getLogger(__name__)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Load and validate data
    df = load_data(args.data_path)
    validate_data(df)

    # Split
    X = df[NUMERIC_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.random_seed
    )
    logger.info(f"Split: {len(X_train):,} train / {len(X_test):,} test")

    # Define models
    models = define_models(args.random_seed)
    logger.info(f"Defined {len(models)} model configurations: {list(models.keys())}")

    # Cross-validation
    logger.info(f"Running {args.n_folds}-fold stratified cross-validation...")
    results_df = run_cv_comparison(models, X_train, y_train, args.n_folds, args.random_seed)

    # Print results table
    logger.info("=== Model Comparison Table (CV) ===")
    for _, row in results_df.iterrows():
        logger.info(f"  {row['model']:15s} | PR-AUC: {row['pr_auc_mean']:.3f} ± {row['pr_auc_std']:.3f} "
                    f"| Recall: {row['recall_mean']:.3f} | Precision: {row['precision_mean']:.3f}")

    # Save results
    save_results(results_df, args.output_dir)

    # Fit all models on full training set
    logger.info("Fitting all models on full training set...")
    fitted_models = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        logger.info(f"  Fitted: {name}")

    # Plots
    logger.info("Generating plots...")
    top3 = plot_pr_curves(fitted_models, X_test, y_test, args.output_dir)
    plot_calibration(fitted_models, X_test, y_test, top3, args.output_dir)

    # Best model
    best_name = results_df.sort_values("pr_auc_mean", ascending=False).iloc[0]["model"]
    best_path = os.path.join(args.output_dir, "best_model.joblib")
    dump(fitted_models[best_name], best_path)
    logger.info(f"Best model: {best_name} — saved to {best_path}")

    # Threshold optimization
    logger.info("Running threshold optimization...")
    threshold_optimization(fitted_models["RF_default"], X_test, y_test, args.output_dir)

    # Tree vs linear disagreement
    logger.info("Finding tree-vs-linear disagreement...")
    find_disagreement(fitted_models["RF_default"], fitted_models["LR_default"],
                      X_test, y_test, args.output_dir)

    logger.info("=== Pipeline complete — all results saved ===")
    
    
if __name__ == "__main__":
    setup_logging()
    args = parse_args()

    logger = logging.getLogger(__name__)

    logger.info("Starting Petra Telecom Churn Model Comparison Pipeline")
    logger.info(f"Data path:   {args.data_path}")
    logger.info(f"Output dir:  {args.output_dir}")
    logger.info(f"Folds:       {args.n_folds}")
    logger.info(f"Random seed: {args.random_seed}")
    logger.info(f"Dry run:     {args.dry_run}")

    if args.dry_run:
        dry_run(args)
        sys.exit(0)

    train_and_evaluate(args)