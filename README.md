# 🧠 Brain Tumor MRI Image Classification

## 📌 Project Overview
This project classifies brain MRI images into four categories using deep learning and transfer learning models. A Streamlit web application is deployed for real-time predictions and visualization.

## 🧠 Classes
- Glioma
- Meningioma
- Pituitary
- No Tumor

## 🚀 Models Used
- Custom CNN
- MobileNetV2
- ResNet50
- InceptionV3 (Best Model)
- EfficientNetB0

## 🧪 Features
- Dataset overview
- Image upload & prediction
- Top-3 predictions with confidence
- Model comparison (accuracy & loss)
- Confusion matrix & classification report
- Grad-CAM visualization
- Data augmentation visualization

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- NumPy, OpenCV
- Streamlit
- Matplotlib, Seaborn

## 📂 Project Structure
brain_tumer_project/
├── app.py
├── utils.py
├── inceptionv3_best.keras
├── history/
├── reports/
├── sample_images/
├── data/
├── README.md
▶️ How to Run

pip install -r requirements.txt
streamlit run app.py

📊 Results

Best Model: InceptionV3

High accuracy achieved using transfer learning

📌 Conclusion

This system helps automate brain tumor classification with explainable AI and an interactive web interface.

👨‍💻 Developed by: Mahendra Pal
Linkdin: 
