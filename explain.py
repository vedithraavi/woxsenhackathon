import lime
import lime.lime_image
import numpy as np
import tensorflow as tf
import cv2
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the trained model
try:
    model = tf.keras.models.load_model("weed_crop_detector.keras")
except OSError:
    print("Error: Model file 'weed_crop_detector.keras' not found.")
    exit()

# Function to preprocess images (Ensure correct input size for model)
def preprocess_image(image_path, target_size=(128, 128)):  # Ensure model-compatible size
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Function to make predictions (Ensure correct input format for LIME)
def predict_fn(images):
    images = np.array(images)  # Convert to numpy array
    return model.predict(images)

# Load and preprocess a sample image (Fix Windows path issue)
image_path = r"C:\Users\Student\OneDrive\Desktop\sem 6\weed_crop_dataset\crop\crop (25).jpeg"  # Use 'r' for raw string
input_image = preprocess_image(image_path)

# Make a prediction
prediction = model.predict(input_image)[0]  # Get first prediction result
class_labels = ["Weed", "Crop"]  # Define labels (Adjust based on your dataset)
predicted_label = class_labels[np.argmax(prediction)]  # Get class with highest probability
print(f"🔍 Model Prediction: {predicted_label} (Confidence: {max(prediction) * 100:.2f}%)")

# Convert image into the correct format for LIME
explainer = lime.lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(
    input_image[0].astype("double"),  # Ensure correct data type
    predict_fn,
    top_labels=2,
    hide_color=0,
    num_samples=1000
)

# Visualize LIME explanation
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],  
    positive_only=False,
    num_features=5,
    hide_rest=False
)

plt.figure(figsize=(8, 8))
plt.imshow(mark_boundaries(temp, mask))
plt.title(f"LIME Explanation for Weed vs. Crop Detection\nPredicted: {predicted_label}")
plt.axis("off")
plt.show()
