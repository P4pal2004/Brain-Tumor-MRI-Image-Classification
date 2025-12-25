import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    layout="wide"
)

IMG_SIZE = 224
CLASSES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# 🔴 FIXED BEST MODEL (DEPLOYMENT SAFE)
BEST_MODEL_NAME = "ResNet50"
BEST_MODEL_PATH = "resnet50_best.keras"   # MUST be in ROOT directory

# ------------------ HELPERS ------------------
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@st.cache_resource
def load_best_model():
    if not os.path.exists(BEST_MODEL_PATH):
        return None
    return load_model(BEST_MODEL_PATH)

# ------------------ SIDEBAR ------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Prediction", "Model Info"]
)

# ======================================================
# 🔮 PREDICTION TAB
# ======================================================
if page == "Prediction":
    st.title("🧠 Brain Tumor MRI Prediction")

    model = load_best_model()

    if model is None:
        st.error("❌ Trained model not found in deployment.")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload Brain MRI Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Uploaded MRI", width=300)

        img_array = preprocess_image(image)
        preds = model.predict(img_array)[0]

        top_indices = preds.argsort()[-3:][::-1]

        with col2:
            st.subheader("📌 Prediction Result")
            st.info(f"Model Used: **{BEST_MODEL_NAME}**")

            for i, idx in enumerate(top_indices, start=1):
                st.write(
                    f"**Top {i}:** {CLASSES[idx]} — "
                    f"{preds[idx] * 100:.2f}%"
                )

# ======================================================
# 📊 MODEL INFO TAB
# ======================================================
elif page == "Model Info":
    st.title("📊 Model Information")

    if os.path.exists(BEST_MODEL_PATH):
        st.success(f"✅ {BEST_MODEL_NAME} model loaded successfully")
        st.markdown("""
        **Model:** ResNet50  
        **Input Size:** 224 × 224 × 3  
        **Classes:** Glioma, Meningioma, Pituitary, No Tumor  
        **Loss Function:** Categorical Crossentropy  
        **Optimizer:** Adam  
        """)
    else:
        st.error("❌ Model file not found in repository")
