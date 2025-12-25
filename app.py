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

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    layout="wide"
)

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
IMG_SIZE = 224
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]

DATASET_PATH = "data/train"
HISTORY_PATH = "history"
REPORTS_PATH = "reports"

# ALL MODELS ARE IN ROOT DIRECTORY
MODELS = {
    "Baseline CNN": "cnn_baseline_best.keras",
    "MobileNetV2": "mobilenetv2_best.keras",
    "ResNet50": "resnet50_best.keras",
    "InceptionV3": "inceptionv3_best.keras",
}

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def preprocess_image(img):
    img = np.array(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)


def load_history_safe(model_key):
    """Safely load history file (cloud-safe)."""
    path = os.path.join(HISTORY_PATH, f"{model_key}_history.npy")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    return None


def get_best_model():
    """Select best model using validation accuracy if history exists."""
    best_model = None
    best_acc = 0

    for name, path in MODELS.items():
        key = name.lower().replace(" ", "_")
        hist = load_history_safe(key)

        if hist and "val_accuracy" in hist:
            acc = max(hist["val_accuracy"])
            if acc > best_acc:
                best_acc = acc
                best_model = name

    # Fallback (Cloud-safe)
    if best_model is None:
        best_model = "ResNet50"

    return best_model


def visualize_augmentation(image, num_aug=4):
    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    img_array = np.expand_dims(np.array(image), axis=0)
    aug_iter = datagen.flow(img_array, batch_size=1)

    return [next(aug_iter)[0].astype(np.uint8) for _ in range(num_aug)]


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Prediction", "Dataset Overview", "Model Comparison"]
)

# ==================================================
# 🔮 PREDICTION TAB
# ==================================================
if page == "Prediction":
    st.title("🧠 Brain Tumor MRI Prediction")

    uploaded_file = st.file_uploader(
        "Upload MRI Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Uploaded MRI", width=350)

        with col2:
            best_model_name = get_best_model()
            model_path = MODELS[best_model_name]

            model = load_model(model_path)
            img_array = preprocess_image(image)
            preds = model.predict(img_array)[0]

            st.subheader("Prediction Result")
            st.info(f"Model Used: **{best_model_name}**")

            top_idx = preds.argsort()[-3:][::-1]
            class_labels = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

            for i, idx in enumerate(top_idx, 1):
                st.write(
                    f"**Top {i}: {class_labels[idx]}** "
                    f"— {preds[idx]*100:.2f}%"
                )

# ==================================================
# 📁 DATASET OVERVIEW
# ==================================================
elif page == "Dataset Overview":
    st.title("📁 Dataset Overview")

    if not os.path.exists(DATASET_PATH):
        st.error("Dataset not found in Streamlit Cloud.")
    else:
        classes = sorted(os.listdir(DATASET_PATH))
        col1, col2 = st.columns(2)

        for i, cls in enumerate(classes):
            cls_path = os.path.join(DATASET_PATH, cls)
            images = os.listdir(cls_path)

            target_col = col1 if i % 2 == 0 else col2

            with target_col:
                st.subheader(cls.capitalize())
                st.write(f"Total Images: **{len(images)}**")

                sample_img = Image.open(
                    os.path.join(cls_path, images[0])
                )
                st.image(sample_img, width=230)

                st.caption("Augmented Samples")
                aug_imgs = visualize_augmentation(sample_img)

                aug_cols = st.columns(len(aug_imgs))
                for c, img in zip(aug_cols, aug_imgs):
                    c.image(img, width=120)

                st.markdown("---")

# ==================================================
# 📈 MODEL COMPARISON
# ==================================================
elif page == "Model Comparison":
    st.title("📈 Model Performance Comparison")

    histories = {}
    for name in MODELS:
        key = name.lower().replace(" ", "_")
        hist = load_history_safe(key)
        if hist:
            histories[name] = hist

    if not histories:
        st.warning("No training history found.")
    else:
        # Accuracy & Loss Curves
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        for name, hist in histories.items():
            ax[0].plot(hist["val_accuracy"], label=name)
            ax[1].plot(hist["val_loss"], label=name)

        ax[0].set_title("Validation Accuracy")
        ax[1].set_title("Validation Loss")
        for a in ax:
            a.legend()

        st.pyplot(fig)

        # Comparison Table
        table = []
        for name, hist in histories.items():
            table.append({
                "Model": name,
                "Best Val Accuracy": max(hist["val_accuracy"]),
                "Min Val Loss": min(hist["val_loss"])
            })

        df = pd.DataFrame(table)
        st.subheader("📊 Comparison Table")
        st.dataframe(df, use_container_width=True)

        st.subheader("🏆 Accuracy Comparison")
        st.bar_chart(df.set_index("Model")["Best Val Accuracy"])

        # Confusion Matrix (Optional)
        y_true_path = os.path.join(REPORTS_PATH, "resnet_y_true.npy")
        y_pred_path = os.path.join(REPORTS_PATH, "resnet_y_pred.npy")

        if os.path.exists(y_true_path) and os.path.exists(y_pred_path):
            y_true = np.load(y_true_path)
            y_pred = np.load(y_pred_path)

            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm, annot=True, fmt="d",
                xticklabels=CLASSES,
                yticklabels=CLASSES
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            st.subheader("📄 Classification Report")
            st.text(
                classification_report(
                    y_true, y_pred,
                    target_names=CLASSES
                )
            )

