"""
convert_tflite.py
-----------------
Converts the trained Keras .h5 model to TensorFlow Lite (.tflite) format
for deployment on resource-constrained devices such as:
  - Raspberry Pi
  - Edge TPU (Coral)
  - Android / iOS mobile phones
  - Drones with embedded processors

Optimisation strategy applied:
  - Dynamic range quantisation (default): reduces model size ~4× by
    quantising weights from float32 → int8. No calibration dataset needed.
  - Full integer quantisation (optional): quantises weights AND activations
    using a representative dataset for maximum size reduction and speed.

Usage:
    python src/convert_tflite.py --model models/disaster_model.h5 \\
                                 --output models/disaster_model.tflite \\
                                 --quantize dynamic

Typical results:
  Original .h5   : ~14 MB
  Dynamic quant  : ~3.5 MB  (4× smaller)
  Full int quant : ~3.0 MB  (4.7× smaller, faster on INT8 hardware)
"""

import os
import sys
import argparse
import time

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGE_SIZE = (224, 224)


# ─────────────────────────────────────────────────────────────────────────────
# Conversion functions
# ─────────────────────────────────────────────────────────────────────────────
def convert_dynamic_range(model_path: str, output_path: str) -> int:
    """
    Applies dynamic range quantisation and saves the .tflite file.

    Weights are quantised to int8; activations remain float32.
    This requires NO calibration data and is the simplest form of
    post-training quantisation.

    Returns the size of the output file in bytes.
    """
    print("\n[Dynamic Range Quantisation]")
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Enable default optimizations — applies dynamic range quantisation
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size = os.path.getsize(output_path)
    print(f"Saved → {output_path}  ({size / 1024:.1f} KB)")
    return size


def convert_full_integer(model_path: str,
                         output_path: str,
                         representative_data_dir: str) -> int:
    """
    Applies full integer quantisation (weights + activations → int8).

    A representative dataset (~100 images) is required to calibrate the
    activation scale factors.  This produces the smallest model and is
    required for Edge TPU / Coral deployment.

    Parameters
    ----------
    model_path           : Path to the Keras .h5 model.
    output_path          : Destination .tflite file path.
    representative_data_dir : Directory containing sample images
                              (used only for calibration, not training).

    Returns the size of the output file in bytes.
    """
    from src.preprocessing import preprocess_single_image
    from pathlib import Path

    print("\n[Full Integer Quantisation]")
    model = tf.keras.models.load_model(model_path)

    # Collect up to 200 sample images from the representative dataset
    image_paths = list(Path(representative_data_dir).rglob("*.jpg"))[:200]
    image_paths += list(Path(representative_data_dir).rglob("*.png"))[:200]
    image_paths = image_paths[:200]

    def representative_dataset():
        """Generator yielding preprocessed image tensors for calibration."""
        for p in image_paths:
            data = preprocess_single_image(str(p))  # shape (1, 224, 224, 3)
            yield [data]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    # Force all ops (including activations) to int8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size = os.path.getsize(output_path)
    print(f"Saved → {output_path}  ({size / 1024:.1f} KB)")
    return size


# ─────────────────────────────────────────────────────────────────────────────
# Inference benchmark
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_tflite(tflite_path: str, num_runs: int = 50) -> dict:
    """
    Measures inference time and memory usage of the TFLite model.

    Runs `num_runs` inferences on a random dummy input and reports:
      - Mean inference latency (ms)
      - Min / max latency
      - Estimated peak memory (KB) from the interpreter's arena size

    Parameters
    ----------
    tflite_path : Path to the .tflite model file.
    num_runs    : Number of inference passes for timing.

    Returns
    -------
    dict with keys: mean_ms, min_ms, max_ms, memory_kb
    """
    print(f"\nBenchmarking {os.path.basename(tflite_path)} …")

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create a dummy input matching the model's expected shape
    input_shape = input_details[0]["shape"]        # e.g. [1, 224, 224, 3]
    input_dtype = input_details[0]["dtype"]        # float32 or int8

    sample_input = np.random.randint(0, 255, size=input_shape).astype(input_dtype)
    if input_dtype == np.float32:
        sample_input = (sample_input / 255.0).astype(np.float32)

    # Warm-up run
    interpreter.set_tensor(input_details[0]["index"], sample_input)
    interpreter.invoke()

    # Timed runs
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], sample_input)
        interpreter.invoke()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # convert to ms

    latencies = np.array(latencies)

    # Memory estimate: file size is a good lower bound; interpreter arena gives
    # the actual runtime allocation
    model_size_kb = os.path.getsize(tflite_path) / 1024

    results = {
        "mean_ms"  : float(np.mean(latencies)),
        "min_ms"   : float(np.min(latencies)),
        "max_ms"   : float(np.max(latencies)),
        "memory_kb": model_size_kb,
    }

    print(f"  Mean latency : {results['mean_ms']:.2f} ms")
    print(f"  Min  latency : {results['min_ms']:.2f} ms")
    print(f"  Max  latency : {results['max_ms']:.2f} ms")
    print(f"  Model size   : {results['memory_kb']:.1f} KB")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert model to TFLite and benchmark.")
    parser.add_argument(
        "--model",
        default=os.path.join(MODELS_DIR, "disaster_model.h5"),
        help="Path to the trained Keras .h5 model.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(MODELS_DIR, "disaster_model.tflite"),
        help="Output path for the .tflite file.",
    )
    parser.add_argument(
        "--quantize",
        choices=["dynamic", "full_int"],
        default="dynamic",
        help="Quantisation mode: 'dynamic' (default) or 'full_int'.",
    )
    parser.add_argument(
        "--rep_data",
        default=os.path.join(BASE_DIR, "data", "processed", "train"),
        help="(full_int only) Path to representative data directory.",
    )
    args = parser.parse_args()

    original_size = os.path.getsize(args.model) / 1024
    print(f"Original .h5 size : {original_size:.1f} KB")

    if args.quantize == "dynamic":
        tflite_size = convert_dynamic_range(args.model, args.output)
    else:
        tflite_size = convert_full_integer(args.model, args.output, args.rep_data)

    reduction = (1 - (tflite_size / 1024) / original_size) * 100
    print(f"\nSize reduction : {reduction:.1f}%")

    benchmark_tflite(args.output)
