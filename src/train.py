"""
train.py
--------
End-to-end training script for the Disaster Detection CNN.

Usage (from the project root):
    python src/train.py

Training strategy:
  Phase 1 — Head training (10 epochs)
    • MobileNetV2 base frozen; only the custom Dense head is trained.
    • Uses a moderate learning rate (1e-3).

  Phase 2 — Fine-tuning (10 epochs, optional)
    • Top 30 backbone layers unfrozen; small LR (1e-5) used to avoid
      catastrophic forgetting.

Callbacks used:
  - ModelCheckpoint : saves the best model (by val_accuracy) to models/
  - EarlyStopping   : halts training if val_loss stops improving for 5 epochs
  - ReduceLROnPlateau: halves LR when val_loss plateaus for 3 epochs
  - TensorBoard     : logs metrics for visualisation (optional)
"""

import os
import sys

# Ensure src/ is on the path when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

from src.preprocessing import split_dataset, get_data_generators
from src.model import build_model, compile_model, unfreeze_top_layers, model_summary

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
LOG_DIR       = os.path.join(BASE_DIR, "logs")

PHASE1_EPOCHS = 2   # Reduced for fast exam training
PHASE2_EPOCHS = 0   # Skip fine-tuning for speed
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3


def get_callbacks(model_name: str = "disaster_model"):
    """
    Builds and returns a list of Keras callbacks.

    Callbacks:
    - ModelCheckpoint  : Saves only the best model weights (val_accuracy).
    - EarlyStopping    : Stops training if val_loss doesn't improve for 5 epochs,
                         then restores the best weights automatically.
    - ReduceLROnPlateau: Halves the LR if val_loss plateaus for 3 epochs.
    - TensorBoard      : Writes training logs for the TensorBoard dashboard.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}.h5")

    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,        # Only overwrite when val_accuracy improves
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,                 # Stop after 5 epochs of no improvement
            restore_best_weights=True,  # Restore weights from the best epoch
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,                 # Multiply LR by 0.5 on plateau
            patience=3,                 # Trigger after 3 stagnant epochs
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=1,
        ),
    ]
    return callbacks


def plot_history(history, title_prefix: str = "Phase 1", save_dir: str = MODELS_DIR):
    """
    Plots and saves accuracy and loss curves for a training history object.

    Parameters
    ----------
    history     : Keras History object.
    title_prefix: Label prepended to the figure title.
    save_dir    : Directory where the plot PNG is saved.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy subplot
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title(f"{title_prefix} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss subplot
    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title(f"{title_prefix} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_dir, f"{title_prefix.replace(' ', '_')}_history.png")
    plt.savefig(fname, dpi=150)
    print(f"Training history plot saved → {fname}")
    plt.show()


def main():
    print("=" * 60)
    print("  Disaster Detection — Training Script")
    print("=" * 60)

    # ── Step 1: Split raw data into train / val / test ─────────────────────
    print("\n[1/5] Splitting dataset…")
    split_dataset(RAW_DIR, PROCESSED_DIR)

    # ── Step 2: Build data generators ─────────────────────────────────────
    print("\n[2/5] Building data generators…")
    train_gen, val_gen, test_gen = get_data_generators(PROCESSED_DIR)
    print(f"      Train batches : {len(train_gen)}")
    print(f"      Val batches   : {len(val_gen)}")
    print(f"      Test batches  : {len(test_gen)}")
    print(f"      Class indices : {getattr(train_gen, 'class_indices', 'Not available')}")

    # ── Step 3: Build and compile the model ────────────────────────────────
    print("\n[3/5] Building model…")
    model = build_model()
    model = compile_model(model, learning_rate=LEARNING_RATE)
    model_summary(model)

    # ── Phase 1: Train only the custom head ───────────────────────────────
    print(f"\n[4/5] Phase 1 Training — {PHASE1_EPOCHS} epochs (head only)…")
    callbacks = get_callbacks("disaster_model")

    history1 = model.fit(
        train_gen,
        epochs=PHASE1_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )
    plot_history(history1, title_prefix="Phase 1 Head Training")

    # ── Phase 2: Fine-tune top backbone layers ─────────────────────────────
    if PHASE2_EPOCHS > 0:
        print(f"\n[5/5] Phase 2 Fine-tuning — {PHASE2_EPOCHS} epochs…")
        model = unfreeze_top_layers(model, num_layers_to_unfreeze=30, fine_tune_lr=1e-5)

        history2 = model.fit(
            train_gen,
            epochs=PHASE2_EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1,
        )
        plot_history(history2, title_prefix="Phase 2 Fine-tuning")
    else:
        print("\n[5/5] Phase 2 skipped (PHASE2_EPOCHS = 0).")

    # Save the final model
    final_path = os.path.join(MODELS_DIR, "disaster_model_final.h5")
    model.save(final_path)
    print(f"\nFinal model saved → {final_path}")

    print("\nTraining complete!")
    return model, test_gen


if __name__ == "__main__":
    main()
