
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Google Drive link of your model
drive_url = "https://drive.google.com/file/d/1Q2YL3PLCuec5-PM_6iNNQyLe6_pCz6ai/view?usp=sharing"

# Output path for model
output_path = "best_model.h5"

# Download using fuzzy mode
if not os.path.exists(output_path):
    print("Downloading model from Drive...")
    gdown.download(drive_url, output_path, quiet=False, fuzzy=True)
    print("Download completed.")
else:
    print("Model already exists.")

# Load model
model = load_model(output_path)
print("Model loaded successfully.")

# Prediction function
def predict_image(img_path, img_size=(224, 224)):
    img = load_img(img_path, target_size=img_size)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = model.predict(arr)[0][0]
    label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    return label, prob


# Example
if __name__ == "__main__":
    img_path = "../dataset/test/PNEUMONIA/person1_virus_6.jpeg"  
    label, prob = predict_image(img_path)
    print(f"Prediction = {label}, Confidence = {prob:.4f}")
