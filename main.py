import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ============================================
# STEP 1: DATA LOADING AND PREPROCESSING
# ============================================

print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values (0-255 -> 0-1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Add channel dimension (batch, height, width, channels)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Training data: {x_train.shape}")
print(f"Testing data: {x_test.shape}")
print(f"Pixel range: [{x_train.min():.2f}, {x_train.max():.2f}]")

# ============================================
# STEP 2: VISUALIZE SAMPLE DATA
# ============================================

plt.figure(figsize=(12, 3))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i].squeeze(), cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.suptitle("Sample MNIST Digits")
plt.tight_layout()
plt.show()


# ============================================
# STEP 3: BUILD SIMPLE CNN MODEL (2 Conv Layers)
# ============================================

def build_simple_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    Simple CNN with 2 Convolutional Layers
    """
    model = models.Sequential([
        # First Convolutional Layer
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # Second Convolutional Layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


# Build and compile report
print("\n" + "=" * 50)
print("BUILDING SIMPLE CNN MODEL")
print("=" * 50)

model = build_simple_cnn()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================
# STEP 4: TRAIN THE MODEL
# ============================================

print("\n" + "=" * 50)
print("TRAINING MODEL")
print("=" * 50)

history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.2,
    verbose=1
)

# ============================================
# STEP 5: EVALUATE THE MODEL
# ============================================

print("\n" + "=" * 50)
print("EVALUATING MODEL")
print("=" * 50)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# ============================================
# STEP 6: VISUALIZE TRAINING HISTORY
# ============================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Loss
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()

plt.show()

# ============================================
# STEP 7: VISUALIZE PREDICTIONS
# ============================================

# Make predictions
predictions = model.predict(x_test[:20], verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

plt.figure(figsize=(15, 3))
for i in range(20):
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap='gray')

    true_label = y_test[i]
    pred_label = predicted_classes[i]
    confidence = predictions[i][pred_label] * 100

    color = 'green' if true_label == pred_label else 'red'
    plt.title(f"T:{true_label} P:{pred_label}\n{confidence:.0f}%",
              color=color, fontsize=8)
    plt.axis('off')

plt.suptitle("Predictions (Green=Correct, Red=Wrong)")
plt.tight_layout()
plt.show()

# ============================================
# STEP 8: CONFUSION MATRIX
# ============================================

y_pred = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# ============================================
# STEP 9: SAVE MODEL
# ============================================

# Save report in models directory
import os

os.makedirs('../models', exist_ok=True)
model.save('cnn_mnist.h5')
print("\n✅ Model saved as '/cnn_mnist.h5'")

# Also save in current directory for easy access
model.save('F:\kushan\Task 2\model')
print("✅ Model also saved in current 'F:\kushan\Task 2\model' directory'' as 'simple_cnn_mnist.h5'")


# ============================================
# STEP 10: CUSTOM IMAGE PREDICTION
# ============================================

def predict_digit(model, image_path):
    """Predict digit from custom image"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return

    # Preprocess same as training
    img = cv2.resize(img, (28, 28))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))

    # Predict
    prediction = model.predict(img, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100

    # Display
    plt.figure(figsize=(4, 4))
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.1f}%")
    plt.axis('off')
    plt.show()

    return predicted_class, confidence


# Usage: predict_digit(report, 'your_image.png')

print("\n" + "=" * 50)
print("✅ Simple CNN Model Complete!")
print("=" * 50)
print("Model Architecture:")
print("  • Conv2D (32 filters)")
print("  • MaxPooling2D")
print("  • Conv2D (64 filters)")
print("  • MaxPooling2D")
print("  • Flatten")
print("  • Dense (128 units)")
print("  • Dropout (0.5)")
print("  • Dense (10 units - output)")