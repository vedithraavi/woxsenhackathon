import tensorflow as tf
from tensorflow.keras.models import Model #ignore
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa  # For Focal Loss
import matplotlib.pyplot as plt
import numpy as np
import os

# Define dataset path
dataset_path = 'weed_crop_dataset'  
img_size = 128
batch_size = 32

# Updated Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)

# Load Training & Validation Data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Load Pretrained ResNet50 Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze the base model layers

# Add Custom Layers on Top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x) 
x = Dropout(0.5)(x) 
x = Dense(1, activation='sigmoid')(x)  # Binary classification (Weed vs Crop)

# Define Final Model
model = Model(inputs=base_model.input, outputs=x)

# Compile Model with Focal Loss (for Imbalanced Dataset)
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss=tfa.losses.SigmoidFocalCrossEntropy(),
              metrics=['accuracy'])

# Show Model Summary
model.summary()

# Train the Model with Early Stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

epochs = 20
history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=[early_stop])

# Save Model
model.save('weed_crop_detector.keras')

# Function to Predict & Display Image
def predict_image(image_path, model):
    if not os.path.exists(image_path):
        print(f"Error: File not found - {image_path}")
        return

    img = load_img(image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    confidence = prediction * 100 if prediction >= 0.5 else (1 - prediction) * 100

    # Display Image & Result
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')

    title = "Weed Detected" if prediction >= 0.5 else "Crop Detected"
    plt.title(f"{title}\nConfidence: {confidence:.2f}%")

    # Confidence Bar
    plt.figure(figsize=(4, 2))
    plt.bar(["Crop", "Weed"], [1 - prediction, prediction], color=['green', 'red'])
    plt.xlabel("Category")
    plt.ylabel("Confidence")
    plt.title("Prediction Confidence")
    plt.ylim(0, 1)
    
    plt.show()

# Test Prediction without Retraining
test_image_path = input("Enter the full path of the test image: ").strip()
predict_image(test_image_path, model)
