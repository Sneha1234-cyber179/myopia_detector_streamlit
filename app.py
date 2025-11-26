# app.py
import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import gdown
import tensorflow as tf
import time

# -----------------------
# Config: Drive model ID
# -----------------------
FILE_ID = "1Ab8usoWhto9Kol8GawbehnsQvwSQjzhB"   # <-- your Drive file id
MODEL_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
MODEL_PATH = "myopia_detection_model.h5"

st.set_page_config(page_title="Myopia Detector", layout="centered")

st.title("👁️ Myopia Detection (Left & Right Eye)")
st.markdown(
    "Capture your **LEFT** eye first then **RIGHT** eye using the camera button below. "
    "Make sure you place only one eye centered in the camera frame."
)

# -----------------------
# Download model if needed
# -----------------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model (may take 30–60s)..."):
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            st.success("Model downloaded.")
        except Exception as e:
            st.error(f"Model download failed: {e}")
            st.stop()

# -----------------------
# Load model
# -----------------------
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# -----------------------
# Helpers
# -----------------------
def pil_to_rgb_array(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB")
    arr = np.asarray(img).astype("float32") / 255.0
    return arr

def prepare_for_model(img_arr: np.ndarray, target_size=(128,128)):
    # Resize with PIL for consistent interpolation
    img = Image.fromarray((img_arr * 255).astype("uint8"))
    img = img.resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0
    # model expects (H,W,3)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    return np.expand_dims(arr, 0)

def classify_from_conf(conf):
    if conf <= 0.3:
        return "NORMAL"
    elif conf >= 0.7:
        return "MYOPIA"
    else:
        return "OTHER / INCONCLUSIVE"

def map_confidence_to_diopter(conf):
    # Example mapping: tune based on your dataset
    diopter = -1.0 - (conf * 5.0)   # approx -1 to -6
    diopter = max(diopter, -12.0)
    return round(diopter, 2)

def predict_from_pil(pil_img):
    arr = pil_to_rgb_array(pil_img)
    X = prepare_for_model(arr, target_size=(128,128))
    pred = model.predict(X)[0][0]
    return float(pred)

# -----------------------
# Session state for flow
# -----------------------
if "left_img" not in st.session_state:
    st.session_state.left_img = None
if "right_img" not in st.session_state:
    st.session_state.right_img = None
if "left_conf" not in st.session_state:
    st.session_state.left_conf = None
if "right_conf" not in st.session_state:
    st.session_state.right_conf = None
if "done" not in st.session_state:
    st.session_state.done = False

# -----------------------
# Capture UI
# -----------------------
st.subheader("1) Capture Eyes (use your browser camera)")
st.info("Capture LEFT eye first. Use the camera button — center ONE eye in the camera frame and capture.")

col1, col2 = st.columns(2)

with col1:
    left_file = st.camera_input("Capture LEFT eye")
    if left_file is not None:
        left_img = Image.open(left_file)
        st.image(left_img, caption="LEFT eye (captured)", use_column_width=True)
        st.session_state.left_img = left_img

with col2:
    right_file = st.camera_input("Capture RIGHT eye")
    if right_file is not None:
        right_img = Image.open(right_file)
        st.image(right_img, caption="RIGHT eye (captured)", use_column_width=True)
        st.session_state.right_img = right_img

# Buttons to predict and reset
st.subheader("2) Scan & Analyze")
col3, col4 = st.columns([1,1])

with col3:
    if st.button("Run Analysis", key="run"):
        if st.session_state.left_img is None or st.session_state.right_img is None:
            st.warning("Please capture both LEFT and RIGHT eye images before analysis.")
        else:
            with st.spinner("Running predictions..."):
                try:
                    lc = predict_from_pil(st.session_state.left_img)
                    rc = predict_from_pil(st.session_state.right_img)
                    st.session_state.left_conf = lc
                    st.session_state.right_conf = rc
                    st.session_state.done = True
                except Exception as e:
                    st.error(f"Prediction error: {e}")
with col4:
    if st.button("Reset", key="reset"):
        st.session_state.left_img = None
        st.session_state.right_img = None
        st.session_state.left_conf = None
        st.session_state.right_conf = None
        st.session_state.done = False
        st.experimental_rerun()

# -----------------------
# Show results if done
# -----------------------
if st.session_state.done:
    st.subheader("3) Results")
    lc = st.session_state.left_conf
    rc = st.session_state.right_conf
    st.markdown(f"- **Left eye confidence (myopia score):** `{lc*100:.1f}%`")
    st.markdown(f"- **Right eye confidence (myopia score):** `{rc*100:.1f}%`")

    # Use worst-case (max) for diagnosis
    max_conf = max(lc, rc)
    classification = classify_from_conf(max_conf)

    if classification == "NORMAL":
        diopter_text = "0.0 D"
        color = "green"
    elif classification == "MYOPIA":
        diopter_text = f"{map_confidence_to_diopter(max_conf)} D"
        color = "red"
    else:
        diopter_text = "-- Consult Specialist --"
        color = "orange"

    st.markdown(f"**Final diagnosis:** <span style='color:{color}'>{classification}</span>", unsafe_allow_html=True)
    st.markdown(f"**Diopter estimate:** `{diopter_text}`")

    # Offer downloadable report (text)
    report = (
        f"--- Eye Diopter Scanner Report ---\n"
        f"Scan Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"----------------------------------\n"
        f"FINAL DIAGNOSIS: {classification}\n"
        f"DIOPTER ESTIMATE: {diopter_text}\n"
        f"----------------------------------\n"
        f"Left Eye Confidence: {lc*100:.1f}%\n"
        f"Right Eye Confidence: {rc*100:.1f}%\n"
        f"----------------------------------\n"
        f"Disclaimer: This is a preliminary computer-generated prediction and does not replace a professional eye examination.\n"
    )
    st.download_button("Download Report (TXT)", report, file_name="myopia_report.txt", mime="text/plain")

    # Show small preview images again
    st.markdown("**Captured images**")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.image(st.session_state.left_img, caption="Left eye", use_column_width=True)
    with pcol2:
        st.image(st.session_state.right_img, caption="Right eye", use_column_width=True)
