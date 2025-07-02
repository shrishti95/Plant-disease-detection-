import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("plant_disease_model.h5")

# Define class labels (replace with actual labels used in training)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
    'Apple___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___healthy'
    # Add all classes here as per your training dataset
]

st.title("🌿 Plant Disease Detection Web App")
st.markdown("Upload an image of a plant leaf to detect the disease.")

# 🟡 This line was missing
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image and ensure it's in RGB mode
    img = Image.open(uploaded_file).convert("RGB")  # 🔧 Force 3 channels

    # Resize to match model input
    img = img.resize((128, 128))

    # Normalize and prepare for prediction
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.image(img, caption='Uploaded Image', use_container_width=True)
    st.success(f"🌿 Predicted Disease: **{predicted_class}**")
