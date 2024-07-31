import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy as np

# Define image size and batch size
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 200

# Prepare data generators
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

def custom_flow_from_directory(directory, batch_size, subset):
    while True:
        if subset == 'training':
            files = os.listdir(os.path.join(directory, 'hand_drawn'))[:int(0.8 * len(os.listdir(os.path.join(directory, 'hand_drawn'))))]
        else:
            files = os.listdir(os.path.join(directory, 'hand_drawn'))[int(0.8 * len(os.listdir(os.path.join(directory, 'hand_drawn')))):]

        batch_hand_drawn = []
        batch_smooth = []
        for file in files:
            hand_drawn_image = cv2.imread(os.path.join(directory, 'hand_drawn', file), cv2.IMREAD_GRAYSCALE)
            smooth_image = cv2.imread(os.path.join(directory, 'smooth', file), cv2.IMREAD_GRAYSCALE)

            hand_drawn_image = np.expand_dims(hand_drawn_image, axis=-1)
            smooth_image = np.expand_dims(smooth_image, axis=-1)

            batch_hand_drawn.append(hand_drawn_image)
            batch_smooth.append(smooth_image)

            if len(batch_hand_drawn) == batch_size:
                yield (np.array(batch_hand_drawn), np.array(batch_smooth))
                batch_hand_drawn = []
                batch_smooth = []

train_generator = custom_flow_from_directory('shapes_dataset', BATCH_SIZE, subset='training')
val_generator = custom_flow_from_directory('shapes_dataset', BATCH_SIZE, subset='validation')

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(IMG_SIZE * IMG_SIZE, activation='sigmoid'),
    layers.Reshape((IMG_SIZE, IMG_SIZE, 1))
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Calculate steps per epoch
num_training_samples = int(0.8 * len(os.listdir('shapes_dataset/hand_drawn')))
num_validation_samples = len(os.listdir('shapes_dataset/hand_drawn')) - num_training_samples

steps_per_epoch = num_training_samples // BATCH_SIZE
validation_steps = num_validation_samples // BATCH_SIZE

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=validation_steps
)

# Save the model
model.save('shape_smoothness_model.h5')
