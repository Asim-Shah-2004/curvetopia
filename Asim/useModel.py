import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained autoencoder model
autoencoder = load_model('autoencoder_best_model.keras')

def load_and_preprocess_image(img_path):
    # Load the image
    img = image.load_img(img_path, color_mode='grayscale', target_size=(128, 128))
    img_array = image.img_to_array(img)
    
    # Normalize the image to range [0, 1]
    img_array = img_array / 255.0
    
    # Reshape the image to match the model's input shape (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def display_image(original, reconstructed):
    plt.figure(figsize=(6, 3))

    # Original image
    ax = plt.subplot(1, 2, 1)
    plt.imshow(original.reshape(128, 128), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Reconstructed image
    ax = plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.reshape(128, 128), cmap='gray')
    plt.title('Reconstructed Image')
    plt.axis('off')

    plt.show()

def reconstruct_image(img_path):
    # Load and preprocess the image
    img_array = load_and_preprocess_image(img_path)
    
    # Predict the reconstructed image using the autoencoder
    reconstructed_img = autoencoder.predict(img_array)
    
    # Display the original and reconstructed images
    display_image(img_array[0], reconstructed_img[0])

# Example usage:
img_path = r'C:\Users\Asim\OneDrive\Desktop\projects\curvetopia\testData\image.png'
reconstruct_image(img_path)
