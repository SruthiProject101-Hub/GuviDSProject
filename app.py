import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model  # ✅ Use standalone keras, not tf.keras

# Load the model once at startup
@st.cache_resource  # ✅ Good caching decorator for models
def load_tb_model():
    model = load_model("vgg16_tb_classifier.h5")  # ✅ Make sure this file exists in your app directory
    return model

model = load_tb_model()

# Function to preprocess uploaded image
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB')  # Ensure 3 color channels
    image = image.resize((224, 224))  # Resize to model input size
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0,1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dim
    return image_array

# Streamlit UI
st.title("🩺 Tuberculosis Chest X-ray Classification")
st.write("Upload a chest X-ray image, and the model will predict if it shows signs of Tuberculosis.")

uploaded_file = st.file_uploader("📤 Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image and predict
        input_array = preprocess_image(image)
        prediction_prob = model.predict(input_array)[0][0]

        # Classification threshold 0.5
        if prediction_prob >= 0.5:
            st.error(f"🛑 Prediction: **Tuberculosis** (Confidence: {prediction_prob:.2f})")
        else:
            st.success(f"✅ Prediction: **Normal** (Confidence: {1 - prediction_prob:.2f})")
    except Exception as e:
        st.error(f"❌ Error processing the image: {e}")
