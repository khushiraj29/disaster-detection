"""
app.py
------
Streamlit web application for the Lightweight CNN-Based Disaster Detection
Framework.

Features:
  - Drag-and-drop or click-to-browse image upload
  - Live inference using either:
      a) Full Keras .h5 model — higher accuracy
      b) TFLite .tflite model — faster, smaller, resource-constrained mode
  - Animated confidence bar chart for all 5 classes
  - Severity indicator and recommended action overlay
  - Inference time display

Run:
    streamlit run app/app.py

Prerequisites:
    pip install streamlit tensorflow pillow numpy
"""

import os
import sys
import time

import numpy as np
from PIL import Image
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
sys.path.insert(0, BASE_DIR)

from src.preprocessing import preprocess_single_image

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["earthquake_damage", "fire", "flood", "landslide", "normal"]

CLASS_META = {
    "earthquake_damage": {
        "emoji"   : "🏚️",
        "severity": "CRITICAL",
        "color"   : "#E74C3C",
        "action"  : "Evacuate the area immediately. Contact emergency services.",
        "details" : "• Drop, cover, and hold on during shaking.\n• If indoors, stay there until the shaking stops.\n• If outdoors, move away from buildings, streetlights, and utility wires.\n• Be prepared for aftershocks.\n• Check for injuries and gas leaks after the quake."
    },
    "fire": {
        "emoji"   : "🔥",
        "severity": "CRITICAL",
        "color"   : "#E67E22",
        "action"  : "Call fire brigade immediately. Do not enter the building.",
        "details" : "• Activate the nearest fire alarm.\n• Evacuate using stairs, never use elevators.\n• If there is smoke, crawl low under it.\n• Check doors for heat before opening.\n• Once out, stay out and gather at the designated assembly area."
    },
    "flood": {
        "emoji"   : "🌊",
        "severity": "HIGH",
        "color"   : "#2980B9",
        "action"  : "Move to higher ground. Avoid contact with floodwater.",
        "details" : "• Evacuate immediately if instructed to do so.\n• Do not walk, swim, or drive through floodwaters—Turn Around, Don't Drown!\n• Stay off bridges over fast-moving water.\n• Move to the highest level of a building if trapped, but do not climb into an enclosed attic."
    },
    "landslide": {
        "emoji"   : "⛰️",
        "severity": "HIGH",
        "color"   : "#8E44AD",
        "action"  : "Evacuate the slope area. Watch for secondary slides.",
        "details" : "• Move away from the path of the landslide or debris flow as quickly as possible.\n• If escape is not possible, curl into a tight ball and protect your head.\n• Listen for unusual sounds that might indicate moving debris, such as trees cracking or boulders knocking together.\n• Be alert for sudden increases or decreases in water flow and for a change from clear to muddy water."
    },
    "normal": {
        "emoji"   : "✅",
        "severity": "NONE",
        "color"   : "#27AE60",
        "action"  : "No disaster detected. Scene appears safe.",
        "details" : "• Continue monitoring if you suspect any impending danger.\n• Ensure emergency kits are easily accessible.\n• Review family emergency plans periodically."
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disaster Detection AI",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS for a polished, dark-themed UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ──────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        color: #e6edf3;
    }

    /* ── Header banner ───────────────────────────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, #1a73e8 0%, #6c35de 100%);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 32px;
        box-shadow: 0 8px 32px rgba(26, 115, 232, 0.3);
    }
    .hero-banner h1 {
        font-size: 2.2rem;
        font-weight: 900;
        color: #ffffff;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        font-size: 1rem;
        color: rgba(255,255,255,0.85);
        margin: 0;
    }

    /* ── Prediction card ─────────────────────────────────────────── */
    .pred-card {
        background: rgba(22, 27, 34, 0.9);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    .pred-class {
        font-size: 3rem;
        font-weight: 900;
        margin: 8px 0 4px 0;
    }
    .pred-confidence {
        font-size: 1.4rem;
        font-weight: 600;
        color: rgba(255,255,255,0.7);
        margin-bottom: 16px;
    }

    /* ── Severity badge ──────────────────────────────────────────── */
    .severity-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 40px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* ── Action box ──────────────────────────────────────────────── */
    .action-box {
        background: rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 18px;
        font-size: 0.95rem;
        color: rgba(255,255,255,0.8);
        text-align: left;
    }

    /* ── Stat pill ───────────────────────────────────────────────── */
    .stat-pill {
        background: rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 12px 18px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stat-pill .stat-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .stat-pill .stat-lbl {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Progress bar override ───────────────────────────────────── */
    div.stProgress > div > div > div > div {
        border-radius: 8px;
    }

    /* ── Sidebar ─────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: rgba(13, 17, 23, 0.95);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached to avoid reloading on every interaction)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_keras_model(path: str):
    """Load and cache the full Keras .h5 model."""
    import tensorflow as tf
    return tf.keras.models.load_model(path)


@st.cache_resource
def load_tflite_model(path: str):
    """Load and cache the TFLite interpreter."""
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter


def run_keras_inference(model, image_array: np.ndarray):
    """
    Runs inference with the full Keras model.

    Parameters
    ----------
    model       : Loaded Keras Model.
    image_array : Preprocessed image of shape (1, 224, 224, 3).

    Returns
    -------
    probs : np.ndarray of shape (5,) — class probabilities.
    ms    : float — inference time in milliseconds.
    """
    t0    = time.perf_counter()
    probs = model.predict(image_array, verbose=0)[0]
    ms    = (time.perf_counter() - t0) * 1000
    return probs, ms


def run_tflite_inference(interpreter, image_array: np.ndarray):
    """
    Runs inference with the TFLite interpreter.

    Parameters
    ----------
    interpreter : TFLite Interpreter (already allocated).
    image_array : Preprocessed image of shape (1, 224, 224, 3).

    Returns
    -------
    probs : np.ndarray of shape (5,) — class probabilities.
    ms    : float — inference time in milliseconds.
    """
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Quantised models expect int8; float models expect float32
    if input_details[0]["dtype"] == np.int8:
        scale, zero_point = input_details[0]["quantization"]
        image_array = (image_array / scale + zero_point).astype(np.int8)

    interpreter.set_tensor(input_details[0]["index"], image_array)

    t0 = time.perf_counter()
    interpreter.invoke()
    ms = (time.perf_counter() - t0) * 1000

    output = interpreter.get_tensor(output_details[0]["index"])[0]

    # De-quantise if output is int8
    if output_details[0]["dtype"] == np.int8:
        scale, zero_point = output_details[0]["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale

    # Apply softmax to convert logits → probabilities (if not already done)
    probs = np.exp(output) / np.sum(np.exp(output)) if output.max() > 1 else output
    return probs, ms


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — model selection & info
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    model_type = st.radio(
        "**Model Backend**",
        options=["Full Keras (.h5)", "TFLite (.tflite) — Lightweight"],
        help="TFLite is faster and smaller; ideal for edge devices.",
    )

    st.markdown("---")
    st.markdown("### 📋 Class Labels")
    for cls in CLASS_NAMES:
        meta = CLASS_META[cls]
        st.markdown(f"{meta['emoji']} `{cls}`")

    st.markdown("---")
    st.markdown("### 🕒 Recent Predictions")
    if "history" in st.session_state and st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            st.caption(f"**{item['time']}** — `{item['class']}` ({item['confidence']:.1f}%)")
    else:
        st.caption("_No predictions yet. Upload an image or use the camera._")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "This tool uses **MobileNetV2** transfer learning to classify "
        "disaster scenes into 5 categories in real time.\n\n"
        "Trained with **TensorFlow/Keras** — optimised for edge deployment."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main page
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🚨 Disaster Detection AI</h1>
    <p>Lightweight CNN — Upload an image to classify the disaster type in real time</p>
</div>
""", unsafe_allow_html=True)

# Input method selection
st.markdown("### 📸 Choose Input Method")
input_method = st.radio("Input Method", ["Upload Image", "Camera"], horizontal=True, label_visibility="collapsed")

if input_method == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a scene image (JPG / PNG / WEBP)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="visible",
    )
else:
    uploaded_file = st.camera_input("Take a picture of the scene")

if uploaded_file is not None:
    # ── Display uploaded image ───────────────────────────────────────────
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("**📷 Uploaded Image**")
        pil_img = Image.open(uploaded_file).convert("RGB")
        st.image(pil_img, use_container_width=True, caption="Input image")

    # ── Save to a temp file so preprocess_single_image can read it ───────
    import tempfile
    suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # ── Preprocess ────────────────────────────────────────────────────────
    image_array = preprocess_single_image(tmp_path)

    # ── Load model & run inference ────────────────────────────────────────
    use_tflite = "TFLite" in model_type

    try:
        with st.spinner("⏳ Analyzing scene... please wait"):
            if use_tflite:
                tflite_path = os.path.join(MODELS_DIR, "disaster_model.tflite")
                if not os.path.exists(tflite_path):
                    st.error(f"TFLite model not found at `{tflite_path}`.\n"
                             "Run `python src/convert_tflite.py` to generate it.")
                    st.stop()
                interpreter = load_tflite_model(tflite_path)
                probs, ms   = run_tflite_inference(interpreter, image_array)
            else:
                keras_path = os.path.join(MODELS_DIR, "disaster_model.h5")
                if not os.path.exists(keras_path):
                    st.error(f"Keras model not found at `{keras_path}`.\n"
                             "Run `python src/train.py` to train the model first.")
                    st.stop()
                model       = load_keras_model(keras_path)
                probs, ms   = run_keras_inference(model, image_array)

        # ── Decode prediction ────────────────────────────────────────────
        pred_idx   = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx]) * 100
        meta       = CLASS_META[pred_class]

        # ── Result card ──────────────────────────────────────────────────
        with col_result:
            st.markdown(f"""
            <div class="pred-card">
                <div style="font-size:3.5rem">{meta['emoji']}</div>
                <div class="pred-class" style="color:{meta['color']}">
                    {pred_class.replace('_', ' ').title()}
                </div>
                <div class="pred-confidence">Confidence: {confidence:.1f}%</div>
                <span class="severity-badge"
                      style="background:{meta['color']}33;color:{meta['color']};
                             border:1px solid {meta['color']};">
                    ⚠️ {meta['severity']} SEVERITY
                </span>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📖 View Actionable Safety Guidelines"):
                st.markdown(f"**Recommended Action:** {meta['action']}")
                st.markdown(f"{meta['details']}")

            # Stats row
            st.markdown("<br>", unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown(f"""
                <div class="stat-pill">
                    <div class="stat-val">{confidence:.1f}%</div>
                    <div class="stat-lbl">Confidence</div>
                </div>""", unsafe_allow_html=True)
            with s2:
                st.markdown(f"""
                <div class="stat-pill">
                    <div class="stat-val">{ms:.1f}ms</div>
                    <div class="stat-lbl">Inference</div>
                </div>""", unsafe_allow_html=True)
            with s3:
                backend_lbl = "TFLite" if use_tflite else "Keras"
                st.markdown(f"""
                <div class="stat-pill">
                    <div class="stat-val">{backend_lbl}</div>
                    <div class="stat-lbl">Backend</div>
                </div>""", unsafe_allow_html=True)

        # ── Probability bar chart ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Confidence per Class")

        for i, (cls, prob) in enumerate(zip(CLASS_NAMES, probs)):
            m   = CLASS_META[cls]
            pct = float(prob) * 100
            is_top = (i == pred_idx)

            label_col, bar_col, pct_col = st.columns([2, 7, 1])
            with label_col:
                weight = "**" if is_top else ""
                st.markdown(f"{m['emoji']} {weight}{cls.replace('_', ' ').title()}{weight}")
            with bar_col:
                st.progress(float(prob))
            with pct_col:
                st.markdown(f"`{pct:.1f}%`")

        # ── Plain text output (as specified in requirements) ──────────────
        st.markdown("---")
        st.info(
            f"🔍 **Detection Result:** "
            f"{'No disaster' if pred_class == 'normal' else pred_class.replace('_', ' ').title()} "
            f"detected – Confidence {confidence:.0f}%"
        )
        
        # Save to history
        if "history" not in st.session_state:
            st.session_state.history = []
        
        # Only add if it's the newest prediction
        current_time = time.strftime("%H:%M:%S")
        if not st.session_state.history or st.session_state.history[-1]['time'] != current_time:
            st.session_state.history.append({
                "time": current_time,
                "class": pred_class.replace('_', ' ').title(),
                "confidence": confidence,
                "backend": backend_lbl
            })
            
        # Download Report
        report_text = (
            "=================================================\n"
            "        DISASTER DETECTION PLATFORM REPORT       \n"
            "=================================================\n\n"
            f"Timestamp      : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Detection      : {pred_class.replace('_', ' ').title()}\n"
            f"Confidence     : {confidence:.2f}%\n"
            "-------------------------------------------------\n"
            f"Severity Level : {meta['severity']}\n"
            f"Action Required: {meta['action']}\n"
            "-------------------------------------------------\n"
            f"Inference Time : {ms:.2f} ms\n"
            f"Backend Engine : {backend_lbl}\n"
            "=================================================\n"
        )
        st.download_button(
            label="📄 Download Detailed Detection Report",
            data=report_text,
            file_name=f"disaster_report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.exception(e)

    finally:
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

else:
    # ── Placeholder when no image is uploaded ─────────────────────────────
    st.markdown("""
    <div style="
        border: 2px dashed rgba(255,255,255,0.2);
        border-radius: 16px;
        padding: 80px 40px;
        text-align: center;
        color: rgba(255,255,255,0.4);
    ">
        <div style="font-size:4rem">📂</div>
        <h3 style="color:rgba(255,255,255,0.5)">Upload an image to get started</h3>
        <p>Supported formats: JPG · PNG · WEBP · BMP</p>
    </div>
    """, unsafe_allow_html=True)
