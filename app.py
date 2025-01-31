import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the pre-trained model (you'll need to save your trained model first)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('breast_tumor_model.h5')
    return model

def preprocess_image(uploaded_file):
    # Read image
    image = Image.open(uploaded_file).convert("RGB")  # Convert to RGB to ensure 3 channels

    # Resize image to match model's expected input
    image = image.resize((256, 256))

    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0

    # Ensure the image has exactly 3 channels
    if img_array.shape[-1] != 3:
        img_array = np.stack((img_array,) * 3, axis=-1)  # Convert grayscale to RGB

    # Add batch dimension (1, 256, 256, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def main():
    st.title('Breast Tumor Classification')

    # Sidebar for model information
    st.sidebar.header('Model Information')
    st.sidebar.write('CNN-based Breast Tumor Classifier')
    st.sidebar.write('Input: Medical Mammogram Images')
    st.sidebar.write('Output: Tumor Classification')

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a Mammogram Image",
        type=["jpg", "jpeg", "png"],
        help="Please upload a medical mammogram image for classification"
    )

    # Prediction section
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption='Uploaded Mammogram', use_column_width=True)

        # Load model
        model = load_model()

        # Preprocess image
        processed_image = preprocess_image(uploaded_file)

        # Make prediction
        prediction = model.predict(processed_image)[0][0]

        # Interpret results
        confidence = prediction * 100
        tumor_type = "Malignant" if prediction > 0.5 else "Benign"

        # Display results with styling
        st.markdown("## Prediction Results")

        # Color-coded result
        if tumor_type == "Malignant":
            st.error(f"🚨 Tumor Classification: {tumor_type}")
        else:
            st.success(f"✅ Tumor Classification: {tumor_type}")

        # Confidence level
        st.metric(
            label="Confidence Level",
            value=f"{confidence:.2f}%",
            help="Percentage confidence in the classification"
        )

        # Additional guidance
        st.warning(
            "**Disclaimer:** This is a screening tool. "
            "Always consult with a medical professional for definitive diagnosis."
        )

if __name__ == "__main__":
    main()