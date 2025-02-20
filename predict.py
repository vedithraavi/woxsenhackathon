import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the trained model
model_path = "weed_crop_detector.keras"
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found. Train the model first.")
    exit()

model = tf.keras.models.load_model(model_path)

# Image size (same as used in training)
IMG_SIZE = 128  

# Function to predict image class
def predict_image(img_path, model):
    if not os.path.exists(img_path):
        print(f"Error: File not found - {img_path}")
        return

    # Load and preprocess image
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Get prediction
    prediction = model.predict(img_array)[0][0]
    confidence = prediction * 100 if prediction >= 0.5 else (1 - prediction) * 100

    # Display image
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')

    # Show prediction result
    title = " Weed Detected" if prediction >= 0.5 else "Crop Detected"
    plt.title(f"{title}\nConfidence: {confidence:.2f}%")

    # Plot confidence bar
    plt.figure(figsize=(4, 2))
    plt.bar(["Crop", "Weed"], [1 - prediction, prediction], color=['green', 'red'])
    plt.xlabel("Category")
    plt.ylabel("Confidence")
    plt.title("Prediction Confidence")
    plt.ylim(0, 1)

    plt.show()

# Ask user for an image file path
test_image_path = input("Enter the full path of the test image: ").strip()
predict_image(test_image_path, model)
