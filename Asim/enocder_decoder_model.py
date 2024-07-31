import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from skimage import filters


#another approach

# Define the autoencoder model
def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    
    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Bottleneck
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    
    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
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

# Paths to dataset folders
hand_drawn_path = '/absolute/path/to/hand_drawn_dataset'
synthetic_path = '/absolute/path/to/synthetic_dataset'

# Load images from both datasets
hand_drawn_images = load_images_from_folder(hand_drawn_path)
synthetic_images = load_images_from_folder(synthetic_path)

# Combine and preprocess images
all_images = np.vstack([hand_drawn_images, synthetic_images])
all_images = all_images.astype('float32') / 255.
all_images = np.reshape(all_images, (len(all_images), 128, 128, 1))

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
train_images, test_images = train_test_split(all_images, test_size=0.2, random_state=42)

# Build and train the autoencoder model
autoencoder = build_autoencoder(input_shape=(128, 128, 1))

# Define model checkpoint callback
checkpoint_path = "autoencoder_best_model.h5"
checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# Train the model
history = autoencoder.fit(train_images, train_images,
                          epochs=50,
                          batch_size=128,
                          shuffle=True,
                          validation_data=(test_images, test_images),
                          callbacks=[checkpoint])

# Load the best model
autoencoder.load_weights(checkpoint_path)

# Evaluate and visualize the results
decoded_imgs = autoencoder.predict(test_images)

def sharpen_image(image):
    return filters.unsharp_mask(image, radius=1, amount=1.5)

n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_images[i].reshape(128, 128), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display smoothed and sharpened
    ax = plt.subplot(2, n, i + 1 + n)
    sharpened_img = sharpen_image(decoded_imgs[i].reshape(128, 128))
    plt.imshow(sharpened_img, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
