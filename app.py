import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Brain Tumor MRI Classification", layout="wide")

IMG_SIZE = 224
CLASSES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

MODELS = {
    "Baseline CNN": "cnn_baseline_best.keras",
    "MobileNetV2": "mobilenetv2_best.keras",
    "ResNet50": "resnet50_best.keras",
    "InceptionV3": "inceptionv3_best.keras",
}

# ---------------- HELPERS ----------------
def preprocess_image(img):
    img = np.array(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def get_available_models():
    return {k: v for k, v in MODELS.items() if os.path.exists(v)}

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Model Info"])

# ==================================================
# 🔮 PREDICTION
# ==================================================
if page == "Prediction":
    st.title("🧠 Brain Tumor MRI Prediction")

    available_models = get_available_models()

    if not available_models:
        st.error("❌ No trained model found in deployment.")
        st.stop()

    model_name = st.selectbox(
        "Select Model",
        list(available_models.keys())
    )

    model = load_model(available_models[model_name])

    uploaded_file = st.file_uploader(
        "Upload MRI Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI", width=300)

        img_array = preprocess_image(image)
        preds = model.predict(img_array)[0]

        top_indices = preds.argsort()[-3:][::-1]

        st.markdown("---")
        st.subheader("Prediction Result")
        st.info(f"**Model Used:** {model_name}")

        for i, idx in enumerate(top_indices, 1):
            st.write(
                f"**Top {i}:** {CLASSES[idx]} — {preds[idx]*100:.2f}%"
            )

# ==================================================
# 📊 MODEL INFO
# ==================================================
elif page == "Model Info":
    st.title("📊 Model Information")

    for name, path in MODELS.items():
        if os.path.exists(path):
            st.success(f"✅ {name} model available")
        else:
            st.warning(f"❌ {name} model not found")

