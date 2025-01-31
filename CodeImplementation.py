##including expected required libraries

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2

# Define paths
malignant_path = "/content/Malignant"
benign_path = "/content/Benign"
image_size = 256
batch_size = 32

# Load dataset
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, target_size=(image_size, image_size))  # Load image
        img_array = img_to_array(img) / 255.0  # Normalize
        images.append(img_array)
        labels.append(label)  # Assign label
    return images, labels
# Load both categories
malignant_images, malignant_labels = load_images_from_folder(malignant_path, label=1)
benign_images, benign_labels = load_images_from_folder(benign_path, label=0)

# Combine and shuffle data
X = np.array(malignant_images + benign_images)
y = np.array(malignant_labels + benign_labels)

# Split dataset (80% Train, 20% Validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

##defining expected convolutional neural network model implementation

def create_custom_model(size=256, dropout_rate=0.5, classes=1):
    model = Sequential()
    # Block 1
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001), input_shape=(size, size, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))  # Early dropout

    # Block 2
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    # Block 3
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.4))

    # Block 4
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.4))

    # Block 5
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.5))  # Strongest dropout here

    # Global Average Pooling for feature reduction
    model.add(GlobalAveragePooling2D())

    # Fully Connected Layers
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Output Layer (Binary classification - Malignant vs. Benign)
    model.add(Dense(classes, activation='sigmoid'))  # Sigmoid for binary classification

    return model

# Initialize Model
model = create_custom_model(size=image_size)

# Compile Model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train Model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# Evaluate Model on Validation Data
val_preds = model.predict(val_dataset)
val_preds = (val_preds > 0.5).astype(int)  # Convert probabilities to binary class

# Generate Classification Report
report = classification_report(y_val, val_preds, target_names=["Benign", "Malignant"])
print(report)