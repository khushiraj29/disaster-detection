"""
clean_images.py
---------------
Scans dataset directories and removes corrupted or truncated images
that would crash training. Verifies images can be fully loaded AND resized.
"""

import os
import gc
from PIL import Image, ImageFile

# Don't allow truncated images to silently pass
ImageFile.LOAD_TRUNCATED_IMAGES = False


def clean_dataset(data_dir):
    """Scan data_dir recursively, delete any image that can't be fully loaded."""
    corrupted = []
    total = 0
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for root, _, files in os.walk(data_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() not in extensions:
                continue
            total += 1
            path = os.path.join(root, f)
            try:
                with Image.open(path) as img:
                    img.load()                    # Force full decode
                    img.resize((224, 224))         # Verify resize works
            except Exception as e:
                print(f"  [CORRUPTED] {path}  —  {e}")
                corrupted.append(path)

    # Release all file handles before deleting
    gc.collect()

    for path in corrupted:
        try:
            os.remove(path)
            print(f"  [DELETED]   {path}")
        except PermissionError:
            print(f"  [SKIP]      {path}  — file locked, delete manually")

    print(f"\nScanned {total} images, removed {len(corrupted)} corrupted files.")


if __name__ == "__main__":
    print("=" * 50)
    print("  Cleaning data/raw/ ...")
    print("=" * 50)
    clean_dataset("data/raw")

    print("\n" + "=" * 50)
    print("  Cleaning data/processed/ ...")
    print("=" * 50)
    clean_dataset("data/processed")

    print("\nDone! You can now re-run: python src/train.py")
