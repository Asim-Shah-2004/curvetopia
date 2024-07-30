import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Ensure TensorFlow uses GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)


def load_labels_from_directory(base_dir):
    """
    Load labels from the specified directory.

    Parameters:
    - base_dir (str): The base directory containing subdirectories for each shape type.

    Returns:
    - labels (list): List of shape labels corresponding to the images.
    """
    labels = []
    shape_types = ['line', 'triangle', 'square', 'circle', 'rectangle', 'star', 'regular_polygon']
    for shape in shape_types:
        shape_dir = os.path.join(base_dir, shape)
        if os.path.isdir(shape_dir):
            for filename in os.listdir(shape_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    labels.append(shape)
    return labels


# Directory path
dataset_directory = '../dataset'

# Load labels and prepare data generators
labels = load_labels_from_directory(dataset_directory)
label_encoder = LabelEncoder()
encoded_labels = to_categorical(label_encoder.fit_transform(labels))

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    dataset_directory,
    target_size=(224, 224),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_directory,
    target_size=(224, 224),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Define the Convolutional Neural Network model
model = Sequential([
    Input(shape=(224, 224, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(224, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(224, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Clear previous TensorFlow sessions
K.clear_session()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model
loss, accuracy = model.evaluate(val_gen)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model.save('../models/shapes_model.keras')
