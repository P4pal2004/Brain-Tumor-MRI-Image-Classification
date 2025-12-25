import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import gdown

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Brain Tumor MRI Classification", layout="wide")

IMG_SIZE = 224
CLASSES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# Google Drive Model
MODEL_URL = "https://drive.google.com/uc?id=1D1L3Ou1W67GXmr_lyR3HG8fkJYWNBLAj"
MODEL_PATH = "inceptionv3_best.keras"

# ------------------ DOWNLOAD MODEL ------------------
@st.cache_resource
def load_best_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading best model (InceptionV3)..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

# ------------------ IMAGE PREPROCESS ------------------
def preprocess_image(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# ------------------ SIDEBAR ------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Model Info"])

# =====================================================
# 🔮 PREDICTION TAB
# =====================================================
if page == "Prediction":
    st.title("🧠 Brain Tumor MRI Prediction")
    st.write("**Best Model Used:** InceptionV3")

    uploaded_file = st.file_uploader(
        "Upload Brain MRI Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Uploaded MRI", width=320)

        with col2:
            model = load_best_model()
            img_array = preprocess_image(image)
            preds = model.predict(img_array)[0]

            top_idx = preds.argsort()[-3:][::-1]

            st.subheader("🔍 Prediction Results")
            for i, idx in enumerate(top_idx, start=1):
                st.write(
                    f"**Top {i}: {CLASSES[idx]}** — {preds[idx]*100:.2f}%"
                )

# =====================================================
# ℹ️ MODEL INFO TAB (VIVA SAFE)
# =====================================================
elif page == "Model Info":
    st.title("📊 Model Information")

    st.markdown("""
    **Model Architecture:** InceptionV3  
    **Input Size:** 224 × 224 × 3  
    **Optimizer:** Adam  
    **Loss Function:** Categorical Crossentropy  
    **Dataset:** Brain MRI Images  
    **Classes:** Glioma, Meningioma, Pituitary, No Tumor  

    ### Why InceptionV3?
    - Deep architecture with factorized convolutions
    - Better feature extraction for medical images
    - High validation accuracy compared to CNN & MobileNet
    """)

    st.success("Model loaded dynamically from Google Drive ✔")

