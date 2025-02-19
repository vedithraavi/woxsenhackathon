import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt

# Set image size
IMG_SIZE = 150  

# Paths for dataset
TRAIN_DIR = r"C:\Users\91630\Downloads\sem6-project\dataset"

# ✅ CHECK DATASET BALANCE
weed_count = len(os.listdir(os.path.join(TRAIN_DIR, "weed")))
no_weed_count = len(os.listdir(os.path.join(TRAIN_DIR, "no_weed")))

print(f"Weed images: {weed_count}, No Weed images: {no_weed_count}")

# ✅ PREPROCESS DATA
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=16,
    class_mode='binary',
    subset="training"
)

val_data = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=16,
    class_mode='binary',
    subset="validation"
)

# ✅ BUILD MODEL
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# ✅ TRAIN MODEL
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)

# ✅ SAVE MODEL
model.save("weed_detector.keras")


# ✅ FUNCTION TO PREDICT & DISPLAY IMAGE
def predict_image(img_path, model):
    if not os.path.exists(img_path):
        print(f"Error: File not found - {img_path}")
        return

    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    prediction = model.predict(img_array)[0][0]
    confidence = prediction * 100 if prediction >= 0.5 else (1 - prediction) * 100

    # ✅ PLOT IMAGE & RESULT
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')

    title = "🌱 Weed Detected" if prediction >= 0.5 else "✅ No Weed Detected"
    plt.title(f"{title}\nConfidence: {confidence:.2f}%")

    # ✅ PLOT CONFIDENCE GRAPH
    plt.figure(figsize=(4, 2))
    plt.bar(["No Weed", "Weed"], [1 - prediction, prediction], color=['green', 'red'])
    plt.xlabel("Category")
    plt.ylabel("Confidence")
    plt.title("Prediction Confidence")
    plt.ylim(0, 1)
    
    plt.show()


# ✅ LOOP TO CHECK MULTIPLE IMAGES
while True:
    test_image_path = input("\nEnter the full path of the test image (or type 'exit' to quit): ").strip()

    if test_image_path.lower() == "exit":
        print("Exiting the program. Goodbye! 👋")
        break

    predict_image(test_image_path, model)
