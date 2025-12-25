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

## 🚀 Deployment Note

Due to GitHub and Streamlit Cloud file size limitations, large deep learning models
(ResNet50, InceptionV3) cannot be uploaded directly.

Therefore:
- The best-performing model (InceptionV3) is hosted on Google Drive
- The Streamlit app dynamically downloads the model at runtime
- This is a standard industry practice for deploying large ML models
- https://drive.google.com/file/d/1D1L3Ou1W67GXmr_lyR3HG8fkJYWNBLAj/view?usp=drive_link

Local execution supports full model comparison and analysis.


📌 Conclusion

This system helps automate brain tumor classification with explainable AI and an interactive web interface.

👨‍💻 Developed by: Mahendra Pal
Linkdin: 
