import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load your model here
# Replace 'your_model_link' with the actual link to your saved model
model = tf.keras.models.load_model('your_model_link')

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    st.title("Alzheimer's Prediction")

    st.write("Upload an MRI scan image for Alzheimer's prediction")

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded MRI.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the uploaded image
        img = preprocess_image(uploaded_file)

        # Make prediction using the model
        prediction = model.predict(img)
        if prediction > 0.5:
            st.write("Prediction: Alzheimer's Disease")
        else:
            st.write("Prediction: Healthy")

if __name__ == "__main__":
    main()
