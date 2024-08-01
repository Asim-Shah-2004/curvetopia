import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the autoencoder model with residual connections
def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    
    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Bottleneck
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    
    # Decoder with residual connections
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Add residual connections
    residual = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(input_img)
    decoded = Add()([decoded, residual])
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

# Load and preprocess the dataset
def load_images_from_folder(folder, img_size=(128, 128)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, target_size=img_size, color_mode='grayscale')
        if img is not None:
            img_array = img_to_array(img)
            images.append(img_array)
    return np.array(images)

# Data Augmentation and Learning Rate Scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return float(lr * tf.math.exp(-0.1))

# Paths to dataset folders
hand_drawn_path = r'C:\Users\Asim\OneDrive\Desktop\projects\curvetopia\Asim\shapes_dataset\hand_drawn'
synthetic_path = r'C:\Users\Asim\OneDrive\Desktop\projects\curvetopia\Asim\shapes_dataset\smooth'

# Load images from both datasets
hand_drawn_images = load_images_from_folder(hand_drawn_path)
synthetic_images = load_images_from_folder(synthetic_path)

# Combine and preprocess images
all_images = np.vstack([hand_drawn_images, synthetic_images])
all_images = all_images.astype('float32') / 255.
all_images = np.reshape(all_images, (len(all_images), 128, 128, 1))

# Split the data into training and testing sets
train_images, test_images = train_test_split(all_images, test_size=0.2, random_state=42)

# Build and train the autoencoder model
autoencoder = build_autoencoder(input_shape=(128, 128, 1))

# Define model checkpoint callback
checkpoint_path = "autoencoder_best_model.keras"
checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# Define data augmentation and learning rate scheduler
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)

lr_scheduler = LearningRateScheduler(scheduler)

# Train the model with augmentation and scheduler
history = autoencoder.fit(datagen.flow(train_images, train_images, batch_size=128),
                          epochs=20,
                          validation_data=(test_images, test_images),
                          callbacks=[checkpoint, lr_scheduler])

# Save the model in .keras format
autoencoder.save('autoencoder_best_model.keras')

# Optionally, plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
