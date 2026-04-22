# Disaster Detection Framework: Project Summary

## 📖 Executive Summary
The **Disaster Detection Framework** is a lightweight, CNN-based artificial intelligence image classification system designed to detect natural disasters from photographs in real-time. Built specifically for resource-constrained edge environments such as drones, mobile phones, and Raspberry Pi devices, the project achieves a high balance between accuracy and computational efficiency.

The model classifies scenes into five distinct categories:
- 🔥 **Fire**
- 🌊 **Flood**
- ⛰️ **Landslide**
- 🏚️ **Earthquake Damage**
- ✅ **Normal** (Safe condition)

## 🏗️ Architecture & Model Strategy
The application utilizes deep neural networks, specifically relying on **Transfer Learning** built on top of the **MobileNetV2** backbone (pre-trained on ImageNet).

- **Two-Phase Training**: Phase 1 freezes the backbone to train a custom Dense classification head. Phase 2 unfreezes the top 30 layers of the feature extractor to fine-tune it for disaster imagery.
- **Data Augmentation**: Enhances training through dynamic, random image rotations, horizontal flips, zooming, and shifting to prevent overfitting.
- **Model Optimization**: Post-training, the framework exports the standard Keras model (`.h5`) into a **TensorFlow Lite** (`.tflite`) package. Utilizing dynamic-range quantization, the model size drops from ~14 MB to ~3.5 MB, enabling extremely fast inference on CPU-bound edge hardware.
- **Cloud Computing Bridge**: Integrated Google Colab workflows (`notebooks/fast_colab_train.ipynb` and `disaster_project_colab.zip`) allow users to seamlessly shift heavy training loads from their local machines to Colab GPUs, reducing training times from several hours to under 10 minutes.

## 🖥️ User Interface Capabilities
A modern, responsive web application acts as the primary frontend for inference:
- **Image & Camera Capture**: Drag-and-drop file support alongside live camera picture-taking capabilities.
- **Dual-Model Inference**: Users can hot-swap between the Full Keras engine or the Lightweight TFLite engine to compare confidence versus inference speeds.
- **Actionable AI**: Provides immediate Severity Level tags (e.g., CRITICAL, HIGH) and displays detailed, context-aware safety recommendations based on the detected disaster.
- **Analytics UI**: Shows a visual animated probability bar chart for all classes, keeping inference times transparently logged on the screen.
- **Reporting**: Features an automatic, downloadable report text file summarising the detection result, confidence, action items, and engine latency.

## 🧰 Tools and Technologies Used

### Machine Learning & Data Processing
- **Python (3.9+)**: The core programming ecosystem powering the system.
- **TensorFlow (2.10+) & Keras**: The primary deep learning framework leveraged for architecture construction, transfer-learning compiling, and gradient descent.
- **TensorFlow Lite**: The optimization toolkit used to quantize the model for mobile and edge deployments.
- **Google Colab / Jupyter Notebooks**: Cloud environments utilized for GPU-accelerated iterative development and model training.
- **Scikit-Learn, Pandas, NumPy**: Standard libraries for splitting training data, array manipulation, and generating sophisticated classification evaluation reports (Precision, Recall, F1-Scores).

### Visualizations & Image Processing
- **Pillow (PIL) & OpenCV**: Handling all raw image ingestion, resizing (to the required 224x224 input), and pixel-array transformations.
- **Matplotlib & Seaborn**: Employed to generate analytical graphs such as annotated heatmaps for confusion matrices.

### Frontend Presentation
- **Streamlit (1.22+)**: Used to rapidly build the dark-themed, interactive, and fully-featured web application without requiring traditional web stack logic.
- **CSS / HTML injection**: Used within the Streamlit constraints to deliver a highly polished UI featuring glassmorphism, gradient badges, and neat layout elements.
