# 🩺 Breast Tumor Classification

## 📌 Overview
This project is a **Streamlit-based web application** that allows users to **upload medical mammogram images** and classifies them as either **"Benign"** or **"Malignant"** using a **Convolutional Neural Network (CNN)** model. The application provides a **user-friendly interface**, displays the **prediction results with confidence levels**, and includes a **disclaimer** to encourage users to consult medical professionals for definitive diagnosis.

---

## ✨ Features
| Feature | Description |
|---------|-------------|
| 🖼️ Upload | Users can upload mammogram images for classification |
| 📊 Prediction | Displays the predicted tumor type (**Benign** or **Malignant**) |
| 📈 Confidence Score | Shows the confidence level of the prediction |
| ⚠️ Disclaimer | Emphasizes the need for **professional medical advice** |
| 🤖 CNN Model | Uses a **pre-trained Convolutional Neural Network** |
| 🌐 Streamlit UI | Provides a **responsive and interactive web application** |

---

## 📚 Dataset
This project used the dataset comprising of mammographic images containing both benign and malignant masses in the female breasts. This dataset was made by extracting 106 mass images from the INbreast, 53 from MIAS and 2,188 from the DDSM dataset. However, my approach only examined the results of the INbreast dataset. The dataset applied data augmentation techniques along with contrast limited adaptive histogram equalization, as a result the number of images increased significantly to 7,632 specifically for the INbreast dataset. To maintain accuracy the dataset contains all images that were resized to 227x227 pixels. It is available to be downloaded open source through: https://data.mendeley.com/datasets/ywsbh3ndr8/2. 
---

## 🏗️ Model Architecture
The **CNN model** used in this project is a custom architecture designed for breast tumor classification. It consists of multiple layers:

- **Convolutional Layers** (Feature Extraction)
- **Batch Normalization** (Stabilization)
- **Max Pooling** (Downsampling)
- **Dropout** (Regularization)
- **Global Average Pooling** (Dimensionality Reduction)
- **Fully Connected Layers** (Classification)

The model is trained on a dataset of **mammogram images**, aiming to accurately distinguish between **benign** and **malignant tumors**.

---

## 🛠️ Installation and Usage
To run the **Breast Tumor Classification** application, follow these steps:

### 1️⃣ Clone the repository:
```bash
git clone https://github.com/Harris-giki/BreastCancerClassification_CNN-DL.git
```

### 2️⃣ Navigate to the project directory:
```bash
cd BreastCancerClassification_CNN-DL
```

### 3️⃣ Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 4️⃣ The pre-trained model is saved with the project directory:
```bash
breast_tumor_model.h5 .
```

### 5️⃣ Run the Streamlit application:
```bash
streamlit run app.py
```

🔗 The application will be available at **http://localhost:8501**. Upload a **mammogram image**, and the app will display the **predicted tumor type** and **confidence level**.

---

## ⚠️ Disclaimer
> **This application is intended for educational and research purposes only.** It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified **healthcare provider** for any medical concerns.

---

## 🤝 Contributions
Contributions are **welcome**! If you have any **suggestions, bug reports**, or want to **add new features**, feel free to:
- **Open an issue** 📝
- **Submit a pull request** 🔄

Let’s build something great together! 🚀
