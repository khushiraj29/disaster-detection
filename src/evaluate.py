"""
evaluate.py
-----------
Evaluation module for the Disaster Detection CNN.

Computes:
  - Accuracy
  - Precision, Recall, F1-Score (per class and weighted average)
  - Confusion Matrix (raw counts and normalised)
  - Classification Report (scikit-learn)

Usage:
    python src/evaluate.py --model models/disaster_model.h5 \\
                           --data  data/processed/test

or imported and called programmatically:
    from src.evaluate import evaluate_model
    evaluate_model(model, test_gen)
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import get_data_generators, CLASS_NAMES

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation function
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model: tf.keras.Model,
                   test_gen,
                   class_names: list = CLASS_NAMES,
                   save_dir: str = MODELS_DIR) -> dict:
    """
    Runs the model on the full test set and computes evaluation metrics.

    Parameters
    ----------
    model       : Trained Keras model.
    test_gen    : Keras DirectoryIterator for the test split.
    class_names : List of class name strings.
    save_dir    : Directory to save confusion matrix plot.

    Returns
    -------
    metrics : dict containing accuracy, precision, recall, f1, report.
    """
    print("\n" + "=" * 60)
    print("  Running Evaluation on Test Set")
    print("=" * 60)

    # ── 1. Collect all true labels and model predictions ──────────────────
    # We reset the generator so we start from the first batch
    test_gen.reset()

    y_true = []
    y_pred_probs = []

    print("Predicting on test set…")
    for i in range(len(test_gen)):
        images, labels = test_gen[i]           # batch of images and one-hot labels
        preds = model.predict(images, verbose=0)  # shape (N, num_classes)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred_probs.extend(preds)

    y_true       = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred       = np.argmax(y_pred_probs, axis=1)  # predicted class index

    # ── 2. Compute top-level metrics ──────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)

    # Weighted averages account for class imbalance
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    print(f"\n{'─'*40}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"{'─'*40}")

    # ── 3. Per-class classification report ────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:\n")
    print(report)

    # ── 4. Confusion matrix ───────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, save_dir=save_dir)

    metrics = {
        "accuracy" : acc,
        "precision": precision,
        "recall"   : recall,
        "f1"       : f1,
        "report"   : report,
        "cm"       : cm,
    }
    return metrics


def plot_confusion_matrix(cm: np.ndarray,
                          class_names: list,
                          save_dir: str = MODELS_DIR) -> None:
    """
    Plots and saves both a raw count and a row-normalised confusion matrix.

    The normalised version (values 0–1) is easier to compare across classes
    of different sizes.

    Parameters
    ----------
    cm          : Raw confusion matrix array from sklearn.
    class_names : List of class labels for axis annotations.
    save_dir    : Directory where the plot is saved as PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Row-normalised (recall per class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)  # Handle zero-rows (missing classes in test)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (Normalised)")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(fname, dpi=150)
    print(f"\nConfusion matrix saved → {fname}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the disaster detection model.")
    parser.add_argument(
        "--model",
        default=os.path.join(MODELS_DIR, "disaster_model.h5"),
        help="Path to the saved .h5 model file.",
    )
    parser.add_argument(
        "--data",
        default=os.path.join(BASE_DIR, "data", "processed"),
        help="Path to the processed data directory (must contain a 'test/' sub-folder).",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model}…")
    model = tf.keras.models.load_model(args.model)

    _, _, test_gen = get_data_generators(args.data)
    evaluate_model(model, test_gen)
