import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Load the trained model
# Ensure the path to the model file is correct
model = load_model("pneumonia_model_vgg16.keras")

st.title("Pneumonia Detection from Chest X-ray Images")
st.write("Upload a chest X-ray image to get a prediction.")

uploaded_file = st.file_uploader("Choose a CXR image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    img = Image.open(uploaded_file).convert("RGB") 
    st.image(img, caption='Uploaded CXR Image.', use_column_width=True)

    # Preprocess the image for the model
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Make prediction
    prediction = model.predict(img_array)

    # Assuming you have the class labels defined somewhere or can get them from the generator
    # For this example, let's assume the class labels are in alphabetical order based on the directory names
    class_labels = ['bacterial', 'covid-19', 'normal', 'viral'] # Adjust if your class labels are different

    # Determine the predicted class label
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]

    # Display the prediction
    st.write(f"Prediction: {predicted_class_label}")
    st.write(f"Prediction probabilities: {prediction[0]}")
