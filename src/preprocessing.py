"""
preprocessing.py
----------------
Handles all data loading, augmentation, and preprocessing steps for the
Lightweight CNN-Based Disaster Detection Framework.

Pipeline:
  1. Scan class directories under `data/raw/`
  2. Split into train / validation / test sets
  3. Apply augmentation on training data (rotation, flip, zoom)
  4. Normalize pixel values to [0, 1]
  5. Return tf.data pipelines ready for model.fit()
"""

import os
import random
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import DirectoryIterator

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE   = (224, 224)   # MobileNetV2 expected input
BATCH_SIZE   = 32
SEED         = 42
CLASS_NAMES  = ["earthquake_damage", "fire", "flood", "landslide", "normal"]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset split utility
# ─────────────────────────────────────────────────────────────────────────────
def split_dataset(raw_dir: str, processed_dir: str,
                  train_ratio: float = 0.70,
                  val_ratio:   float = 0.15,
                  test_ratio:  float = 0.15,
                  seed: int = SEED) -> None:
    """
    Copies images from `raw_dir/<class>/` into
    `processed_dir/{train,val,test}/<class>/` according to the given ratios.

    Parameters
    ----------
    raw_dir      : Path to the raw data folder containing one sub-dir per class.
    processed_dir: Destination root directory for the split dataset.
    train_ratio  : Fraction of images for training.
    val_ratio    : Fraction of images for validation.
    test_ratio   : Fraction of images for testing.
    seed         : Random seed for reproducibility.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        "Ratios must sum to 1."

    random.seed(seed)
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)

    for class_name in CLASS_NAMES:
        class_dir = raw_path / class_name
        if not class_dir.exists():
            print(f"[WARNING] Class directory not found: {class_dir}")
            continue

        # Collect all image files (common extensions)
        images = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
        random.shuffle(images)

        n       = len(images)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        splits = {
            "train": images[:n_train],
            "val":   images[n_train : n_train + n_val],
            "test":  images[n_train + n_val :],
        }

        for split, files in splits.items():
            dest = proc_path / split / class_name
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dest / f.name)

        print(f"[{class_name}] total={n}  "
              f"train={len(splits['train'])}  "
              f"val={len(splits['val'])}  "
              f"test={len(splits['test'])}")

    print("\nDataset split complete.")


# ─────────────────────────────────────────────────────────────────────────────
# ImageDataGenerators
# ─────────────────────────────────────────────────────────────────────────────
def get_data_generators(processed_dir: str) -> tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator]:
    """
    Creates Keras ImageDataGenerators for train, validation, and test sets.

    Augmentation is applied only to training images to improve generalisation
    and reduce over-fitting on a small dataset:
      - Random horizontal flip
      - Random rotation ± 20°
      - Random zoom 10 %
      - Random width/height shift 10 %

    All splits are normalised to [0, 1] by rescaling by 1/255.

    Returns
    -------
    train_gen, val_gen, test_gen : DirectoryIterator objects
    """
    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,           # Normalise pixel values to [0, 1]
        rotation_range=20,           # Randomly rotate images up to 20 degrees
        width_shift_range=0.10,      # Shift width by up to 10 % of the width
        height_shift_range=0.10,     # Shift height by up to 10 % of the height
        shear_range=0.10,            # Shear transformation
        zoom_range=0.10,             # Random zoom up to 10 %
        horizontal_flip=True,        # Random horizontal flip
        fill_mode="nearest",         # Fill newly created pixels
    )

    # Validation & test generators — only normalise, no augmentation
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    proc_path = Path(processed_dir)

    train_gen = train_datagen.flow_from_directory(
        str(proc_path / "train"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
        seed=SEED,
    )

    val_gen = eval_datagen.flow_from_directory(
        str(proc_path / "val"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_directory(
        str(proc_path / "test"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────────────────────────────────────
# Single-image preprocessing (for inference)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_single_image(image_path: str) -> np.ndarray:
    """
    Loads and preprocesses one image for inference.

    Steps:
      1. Load image from disk and convert to RGB
      2. Resize to IMAGE_SIZE (224×224)
      3. Convert to float32 numpy array
      4. Normalise to [0, 1]
      5. Add batch dimension → shape (1, 224, 224, 3)

    Parameters
    ----------
    image_path : Absolute or relative path to the image file.

    Returns
    -------
    np.ndarray of shape (1, 224, 224, 3), dtype float32.
    """
    img = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    arr = tf.keras.utils.img_to_array(img)  # shape (224, 224, 3)
    arr = arr / 255.0                        # normalise
    arr = np.expand_dims(arr, axis=0)        # add batch dim → (1, 224, 224, 3)
    return arr.astype(np.float32)
