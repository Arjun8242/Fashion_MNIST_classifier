import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Load the model
with open("model.pkl", "rb") as f:
    model_params = pickle.load(f)

# Extract model parameters
W1 = model_params["W1"]
b1 = model_params["b1"]
W2 = model_params["W2"]
b2 = model_params["b2"]
W3 = model_params["W3"]
b3 = model_params["b3"]
input_size = model_params["input_size"]
hidden_size1 = model_params["hidden_size1"]
hidden_size2 = model_params["hidden_size2"]
output_size = model_params["output_size"]


class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Activation functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # For numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Function to make predictions
def predict(X):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)
    return np.argmax(A3, axis=1)

# Streamlit app
st.title("Neural Network Deployment with Streamlit")
st.write("Upload an image (28x28 pixels) to classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image).reshape(1, 784)  # Flatten to 1x784

    # Normalize the image (if necessary)
    image_array = image_array / 255.0

    # Make a prediction
    prediction = predict(image_array)
    predicted_class = class_names[prediction[0]]  # Map numeric label to class name

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Display the prediction
    st.write(f"Prediction: {predicted_class}")