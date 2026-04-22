# Lightweight CNN-Based Disaster Detection Framework
## For Resource-Constrained Environments

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange?logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22%2B-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A **transfer-learning CNN** built on **MobileNetV2** that can detect five
disaster types from photographs in real time — optimised for deployment on
Raspberry Pi, drones, and mobile phones via **TensorFlow Lite**.

---

## 🌟 Features

| Feature | Detail |
|---|---|
| **Model** | MobileNetV2 + custom head (transfer learning) |
| **Classes** | Fire · Flood · Landslide · Earthquake damage · Normal |
| **Input size** | 224 × 224 px |
| **Framework** | TensorFlow / Keras |
| **Optimisation** | TFLite dynamic-range quantisation (~4× size reduction) |
| **UI** | Streamlit web app with confidence chart |
| **Output** | "Fire detected – Confidence 93%" |

---

## 📂 Project Structure

```
disaster_detection/
│
├── data/
│   ├── raw/                   ← Place raw images here, one folder per class
│   │   ├── earthquake_damage/
│   │   ├── fire/
│   │   ├── flood/
│   │   ├── landslide/
│   │   └── normal/
│   └── processed/             ← Auto-generated: train / val / test splits
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/
│   ├── disaster_model.h5      ← Best Keras model (saved by ModelCheckpoint)
│   ├── disaster_model_final.h5← Final model after fine-tuning
│   └── disaster_model.tflite  ← Quantised TFLite model for edge devices
│
├── notebooks/
│   └── training.ipynb         ← Interactive Jupyter training walkthrough
│
├── src/
│   ├── preprocessing.py       ← Dataset split, augmentation, single-image prep
│   ├── model.py               ← MobileNetV2 model builder & fine-tuning utils
│   ├── train.py               ← End-to-end training script (Phase 1 + 2)
│   ├── evaluate.py            ← Metrics & confusion matrix generator
│   └── convert_tflite.py      ← TFLite conversion & inference benchmark
│
├── app/
│   └── app.py                 ← Streamlit web application
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/yourname/disaster_detection.git
cd disaster_detection
pip install -r requirements.txt
```

### 2. Prepare your dataset

Place images inside `data/raw/` with one sub-directory per class:

```
data/raw/
    ├── fire/               (e.g. 500+ images)
    ├── flood/
    ├── landslide/
    ├── earthquake_damage/
    └── normal/
```

> **Tip:** Public datasets to get started:
> - [AIDER: Aerial Image Dataset for Emergency Response](https://github.com/ckyrkou/AIDER)
> - [Natural Disaster Images on Kaggle](https://www.kaggle.com/datasets)

### 3. Train the model

```bash
python src/train.py
```

This will:
1. Automatically split the dataset (70% train / 15% val / 15% test)
2. Run **Phase 1** — train only the custom Dense head (10 epochs)
3. Run **Phase 2** — fine-tune top 30 MobileNetV2 layers (10 epochs)
4. Save the best checkpoint to `models/disaster_model.h5`

### 4. Evaluate the model

```bash
python src/evaluate.py --model models/disaster_model.h5 \
                       --data  data/processed
```

Outputs:
- Accuracy, Precision, Recall, F1-Score
- Per-class classification report
- `models/confusion_matrix.png`

### 5. Convert to TFLite

```bash
python src/convert_tflite.py \
    --model    models/disaster_model.h5 \
    --output   models/disaster_model.tflite \
    --quantize dynamic
```

Outputs latency benchmark results for your local machine.

### 6. Run the web application

```bash
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧠 Model Architecture

```
Input (224 × 224 × 3)
        │
MobileNetV2 backbone
  • Depthwise separable convolutions
  • Pretrained on ImageNet (frozen in Phase 1)
  • Top 30 layers unfrozen in Phase 2 fine-tuning
        │
GlobalAveragePooling2D   ← Spatial → feature vector
        │
Dense(256, relu) + Dropout(0.4) + L2 regularisation
        │
Dense(128, relu) + Dropout(0.3) + L2 regularisation
        │
Dense(5, softmax)        ← Output: class probabilities
```

**Why MobileNetV2?**
- Depthwise-separable convolutions use ~8–9× fewer multiply-adds than standard convolutions.
- ~3.4M parameters total (vs. ~25M for VGG-16).
- Achieves 72.0% top-1 accuracy on ImageNet — excellent for a model this small.

---

## 🏋️ Training Strategy

| Phase | Epochs | Frozen Layers | LR | Purpose |
|---|---|---|---|---|
| Phase 1 | 10 | All backbone | 1e-3 | Train custom head |
| Phase 2 | 10 | All except top 30 | 1e-5 | Domain fine-tuning |

**Augmentation (training only):**
- Random rotation ±20°
- Random horizontal flip
- Random zoom ±10%
- Width/height shift ±10%

**Callbacks:**
- `ModelCheckpoint` — saves best `val_accuracy` model
- `EarlyStopping` — patience 5, restores best weights
- `ReduceLROnPlateau` — halves LR after 3 stagnant epochs

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Fraction of correctly classified images |
| **Precision** | Of predicted positives, how many are truly positive |
| **Recall** | Of actual positives, how many were detected |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **Confusion Matrix** | Raw counts + row-normalised heatmap |

---

## ⚡ Model Optimisation (Edge Deployment)

| Format | Size | Notes |
|---|---|---|
| `.h5` (float32) | ~14 MB | Full accuracy, GPU/CPU |
| `.tflite` dynamic | ~3.5 MB | 4× smaller, CPU only |
| `.tflite` full-int | ~3.0 MB | Fastest on INT8 hardware |

**Deployment targets:**
- 🍓 **Raspberry Pi 4** — TFLite runtime, ~200–400 ms/frame
- 📱 **Android/iOS** — ML Kit / Core ML via TFLite bridge
- 🚁 **Drone** — Embedded Linux + TFLite, runs headlessly
- 🔬 **Edge TPU / Coral** — Full int8 quantisation required

---

## 🖥️ Streamlit App

Upload any image and the app will:

1. Preprocess it to 224×224
2. Run inference via the selected backend (Keras or TFLite)
3. Display:
   - Predicted disaster class with emoji
   - Confidence percentage
   - Severity level (NONE / HIGH / CRITICAL)
   - Recommended action
   - Per-class confidence bar chart
   - Inference time

**Example output:**

```
🔥 Fire detected – Confidence 93%
⚠️ CRITICAL SEVERITY
Recommended Action: Call fire brigade immediately. Do not enter the building.
```

---

## 📋 Requirements

```
tensorflow>=2.10.0
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.1.0
matplotlib>=3.6.0
seaborn>=0.12.0
Pillow>=9.0.0
opencv-python>=4.6.0
streamlit>=1.22.0
```

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Acknowledgements

- MobileNetV2: [Howard et al., 2018](https://arxiv.org/abs/1801.04381)
- TensorFlow Lite: [tensorflow.org/lite](https://www.tensorflow.org/lite)
- Streamlit: [streamlit.io](https://streamlit.io)
