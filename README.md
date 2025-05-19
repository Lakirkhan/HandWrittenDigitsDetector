
# 🧠 MNIST Digit Recognizer with CNN and Streamlit

This project demonstrates a simple **Convolutional Neural Network (CNN)** for recognizing handwritten digits using the **MNIST** dataset. The trained model is integrated into a **Streamlit** web application where you can upload your own digit image and get predictions.

---

## 📁 Project Structure
├── mnist_cnn.py # Script to train and save the CNN model
├── mnist_streamlit.py # Streamlit app for digit prediction
├── mnist_cnn_model.h5 # Saved CNN model after training (generated)
└── README.md # Project documentation



---

## ⚙️ Requirements

Install the necessary packages using pip:

```bash
pip install tensorflow streamlit pillow numpy


 How to Run
1. Train the CNN model
Run this script to train and save the model:

python mnist_cnn.py

This will:Load the MNIST dataset

Build and train a CNN

Save the model as mnist_cnn_model.h5

2. Start the Streamlit app
streamlit run mnist_streamlit.py


Upload a 28x28 pixel grayscale handwritten digit image, and the app will:
Preprocess the image
Predict the digit using your trained CNN model
Show the result
