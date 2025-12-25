import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

st.set_page_config(page_title="Brain Tumor MRI Classification", layout="wide")

# ------------------ CONSTANTS ------------------
IMG_SIZE = 224
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
DATASET_PATH = "data/train"
HISTORY_PATH = "history"
REPORTS_PATH = "reports"

MODELS = {
    "Baseline CNN": "cnn_baseline_best.keras",
    "MobileNetV2": "mobilenetv2_best.keras",
    "ResNet50": "resnet50_best.keras",
    "InceptionV3": "inceptionv3_best.keras",
}

# ------------------ HELPERS ------------------
def preprocess_image(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def load_history_safe(name):
    path = os.path.join(HISTORY_PATH, f"{name}_history.npy")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    return None

def get_best_model():
    best_model, best_acc = None, 0
    for name in MODELS:
        hist = load_history_safe(name.lower().replace(" ", "_"))
        if hist and "val_accuracy" in hist:
            acc = max(hist["val_accuracy"])
            if acc > best_acc:
                best_acc = acc
                best_model = name
    return best_model, best_acc

def visualize_augmentation(image, num_aug=4):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    img_array = np.expand_dims(np.array(image), 0)
    aug_iter = datagen.flow(img_array, batch_size=1)
    return [next(aug_iter)[0].astype(np.uint8) for _ in range(num_aug)]

# ------------------ SIDEBAR ------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Dataset Overview", "Model Comparison"])

# ======================================================
# 🔮 PREDICTION TAB
# ======================================================
if page == "Prediction":
    st.title("🧠 Brain Tumor Prediction")

    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width=300)

        best_model_name, best_acc = get_best_model()

        if best_model_name is None:
            st.error("No trained model found in deployment.")
        else:
            model_path = MODELS[best_model_name]
            if not os.path.exists(model_path):
                st.error(f"Model file missing: {model_path}")
            else:
                model = load_model(model_path)
                img_array = preprocess_image(image)
                preds = model.predict(img_array)[0]

                top3 = preds.argsort()[-3:][::-1]
                st.success(f"Best Model Used: {best_model_name}")

                for i, idx in enumerate(top3, 1):
                    st.write(f"Top {i}: **{CLASSES[idx]}** — {preds[idx]*100:.2f}%")

# ======================================================
# 📁 DATASET OVERVIEW
# ======================================================
elif page == "Dataset Overview":
    st.title("📁 Dataset Overview")

    if not os.path.exists(DATASET_PATH):
        st.warning("Dataset not uploaded to GitHub (expected for deployment).")
    else:
        for cls in sorted(os.listdir(DATASET_PATH)):
            cls_path = os.path.join(DATASET_PATH, cls)
            images = os.listdir(cls_path)
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader(cls)
                st.write(f"Total Images: {len(images)}")

            with col2:
                img = Image.open(os.path.join(cls_path, images[0]))
                st.image(img, width=220)

                st.write("Augmented Samples:")
                aug_cols = st.columns(4)
                for c, aug in zip(aug_cols, visualize_augmentation(img)):
                    c.image(aug, width=120)

            st.markdown("---")

# ======================================================
# 📈 MODEL COMPARISON
# ======================================================
elif page == "Model Comparison":
    st.title("📈 Model Comparison")

    histories = {
        name: load_history_safe(name.lower().replace(" ", "_"))
        for name in MODELS
    }

    valid_histories = {k: v for k, v in histories.items() if v}

    if not valid_histories:
        st.warning("No training history files found.")
    else:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        for name, hist in valid_histories.items():
            ax[0].plot(hist["val_accuracy"], label=name)
            ax[1].plot(hist["val_loss"], label=name)

        ax[0].set_title("Validation Accuracy")
        ax[1].set_title("Validation Loss")
        for a in ax: a.legend()
        st.pyplot(fig)

                )
            )

