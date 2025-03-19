import tensorflow as tf
from PIL import Image
import cv2
import os
import numpy as np

# Load the Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Create a directory to save the images
output_dir = 'fashion_mnist_test_images'
os.makedirs(output_dir, exist_ok=True)

# Save images using PIL (Pillow)
for i, (image, label) in enumerate(zip(x_test, y_test)):
    # Convert the NumPy array to a PIL image
    img = Image.fromarray(image, mode='L')  # 'L' mode for grayscale

    # Create a subdirectory for each label (optional)
    label_dir = os.path.join(output_dir, f'label_{label}')
    os.makedirs(label_dir, exist_ok=True)

    # Save the image using PIL
    img.save(os.path.join(label_dir, f'test_image_{i}_pil.png'))

# Save images using OpenCV
for i, (image, label) in enumerate(zip(x_test, y_test)):
    # Create a subdirectory for each label (optional)
    label_dir = os.path.join(output_dir, f'label_{label}')
    os.makedirs(label_dir, exist_ok=True)

    # Save the image using OpenCV
    cv2.imwrite(os.path.join(label_dir, f'test_image_{i}_opencv.png'), image)

print(f"All test images saved to {output_dir}")