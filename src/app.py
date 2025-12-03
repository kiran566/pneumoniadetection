import streamlit as st
import os
import gdown
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

MODEL_FILENAME = "best_model.h5"
MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1Q2YL3PLCuec5-PM_6iNNQyLe6_pCz6ai"

MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    gdown.download(MODEL_DRIVE_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded!")

@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

def predict_image(img, img_size=(224, 224)):
    img = img.convert("RGB")
    img = img.resize(img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prob = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    return label, prob

st.title("🩺 Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image to detect **Pneumonia**.")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded Chest X-ray", use_column_width=True)

    if st.button("Predict"):
        label, confidence = predict_image(img)
        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence:.4f}")
