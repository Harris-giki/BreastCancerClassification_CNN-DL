# Breast Tumor Classification

## Overview
This project is a **Streamlit-based web application** that enables users to upload medical mammogram images and classifies them as either **"Benign" or "Malignant"** using a **convolutional neural network (CNN)** model. The application provides a user-friendly interface, displays prediction results with confidence levels, and includes a disclaimer encouraging users to consult medical professionals for definitive diagnoses.

## Features
- Upload medical mammogram images for classification.
- Displays the uploaded image and the predicted tumor type (**Benign or Malignant**).
- Shows the **confidence level** of the prediction.
- Includes a **disclaimer** emphasizing the need for professional medical advice.
- Utilizes a **pre-trained CNN model** for tumor classification.
- Built with **Streamlit** to create a responsive and interactive web application.

## Model Architecture
The CNN model used in this project is a **custom architecture** designed specifically for breast tumor classification. The model consists of:
- Multiple **convolutional layers** for feature extraction.
- **Batch normalization** to stabilize training.
- **Max pooling layers** to reduce spatial dimensions.
- **Dropout layers** to prevent overfitting.
- A **global average pooling layer** and **fully connected layers** for classification.

The model is trained on a dataset of mammogram images to accurately distinguish between benign and malignant tumors.

## Installation and Usage
To run the Breast Tumor Classification application, follow these steps:

### 1. Clone the repository:
```bash
git clone https://github.com/your-username/breast-tumor-classification.git
```

### 2. Navigate to the project directory:
```bash
cd breast-tumor-classification
```

### 3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 4. Save the pre-trained model to the project directory:
```bash
mv breast_tumor_model.h5 .
```

### 5. Run the Streamlit application:
```bash
streamlit run app.py
```

The application will be available at **http://localhost:8501**. Upload a mammogram image, and the application will display the predicted tumor type and confidence level.

## Disclaimer
This application is intended for **educational and research purposes only**. It should **not** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any medical concerns.

## Contributions
Contributions to this project are **welcome**! If you have suggestions, bug reports, or would like to add new features, feel free to **open an issue** or submit a **pull request**.
