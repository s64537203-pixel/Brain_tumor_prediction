# 🧠 Brain Tumor Detection using Deep Learning

## 📌 Project Overview

This project is a deep learning-based web application that detects brain tumors from MRI images using a pretrained ResNet18 model. Users can upload an MRI scan, and the model predicts the type of brain tumor with high accuracy. The application is designed to demonstrate the practical use of computer vision in medical image classification.

---

## 🚀 Features

* Upload brain MRI images
* Predict tumor type using a trained ResNet18 model
* Supports four classes:

  * Glioma
  * Meningioma
  * Pituitary Tumor
  * No Tumor
* Displays prediction results in a simple and user-friendly interface
* Built with PyTorch and Python

---

## 🧠 Model

* **Architecture:** ResNet18 (Transfer Learning)
* **Framework:** PyTorch
* **Loss Function:** CrossEntropyLoss
* **Optimizer:** Adam
* **Technique:** Transfer Learning

---

## 📂 Dataset

**Brain Tumor MRI Dataset**

Classes:

* Glioma
* Meningioma
* Pituitary
* No Tumor

The dataset contains labeled MRI images for multiclass brain tumor classification.

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Torchvision
* OpenCV
* NumPy
* Pillow
* streamlit

---

## 📁 Project Structure

```text
Brain-Tumor-Detection/
│
├── app.py
├── predict.py
├── best_model.pth
├── requirements.txt
├── README.md
├── templates/
├── static/
└── uploads/
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/s64537203-pixel/Brain-Tumor-Detection.git
```

### Move into the project folder

```bash
cd Brain-Tumor-Detection
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python app.py
```

Open the browser and visit:

```text
http://127.0.0.1:5000
```

---

## 📊 Model Performance

* Training Accuracy: **90%+**
* Multiclass Classification
* Transfer Learning with ResNet18

> *Performance may vary depending on the dataset split and training configuration.*

---

## 📸 Application Preview

Add screenshots here after uploading them.

* Home Page
* MRI Upload
* Prediction Result

---

## 👩‍💻 Author

**Sheetal**

B.Tech Artificial Intelligence & Machine Learning (AI & ML)

Passionate about Deep Learning, Computer Vision, and AI-powered healthcare solutions.

---

## ⭐ If you found this project useful

Please consider giving this repository a ⭐ on GitHub.
