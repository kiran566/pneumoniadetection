import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -----------------------------
# Load the trained model
# -----------------------------
@st.cache_resource
def load_cnn_model():
    return load_model("best_model.h5")

model = load_cnn_model()

# -----------------------------
# Prediction function
# -----------------------------
def predict_image(img, img_size=(224, 224)):
    img = img.resize(img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prob = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    return label, prob

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("🩺 Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image to detect **Pneumonia** using your trained CNN model.")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = load_img(uploaded_file)
    st.image(img, caption="Uploaded Chest X-ray", use_column_width=True)

    # Predict Button
    if st.button("Predict"):
        label, confidence = predict_image(img)
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: `{confidence:.4f}`")

        # Extra UI coloring
        if label == "PNEUMONIA":
            st.error("⚠ Pneumonia Detected. Please consult a doctor.")
        else:
            st.success("✔ Normal Chest X-ray")

st.markdown("---")
st.write("Developed by Kiran – Pneumonia Detection CNN Model")
