# save as mnist_streamlit.py

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

st.title("MNIST Digit Recognizer")

model = load_model('mnist_cnn_model.h5')

uploaded_file = st.file_uploader("Upload a handwritten digit image (28x28 px)")

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')  # grayscale
    image = ImageOps.invert(image)
    image = image.resize((28,28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1,28,28,1)
    
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)
    
    st.image(image, caption="Uploaded Image")
    st.write(f"Predicted Digit: {digit}")
