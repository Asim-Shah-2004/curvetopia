import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = (224, 224)
MODEL_PATH = './star_model2.keras'
IMAGE_PATH = './Untitled.png'  # Replace with your image path

# Load the trained model
model = load_model(MODEL_PATH)


def preprocess_image(img_path, target_size):
    """
    Load and preprocess the image for prediction.

    Parameters:
    - img_path (str): Path to the image file.
    - target_size (tuple): Target size for the image.

    Returns:
    - img_array (numpy.ndarray): Preprocessed image array.
    """
    img = image.load_img(img_path, target_size=target_size, color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale pixel values to [0, 1]
    return img_array


# Preprocess the image
img_array = preprocess_image(IMAGE_PATH, IMAGE_SIZE)

# Make a prediction
prediction = model.predict(img_array)

# Print the prediction
if prediction[0] > 0.5:
    print("The image is classified as a star.")
else:
    print("The image is classified as non-star.")

# Optionally, display the image
img = image.load_img(IMAGE_PATH, color_mode='grayscale')
plt.imshow(img, cmap='gray')
plt.title("Predicted: Star" if prediction[0] > 0.5 else "Predicted: Non-Star")
plt.show()
