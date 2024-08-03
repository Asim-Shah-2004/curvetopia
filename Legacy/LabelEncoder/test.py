import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Define shape labels
shape_labels = ['line', 'circle', 'ellipse', 'square', 'rectangle', 'polygon', 'star']

# Load the trained model
model = load_model('shapes_model.keras')
print(model.summary())


def load_and_preprocess_image(img_path):
    # Load the image
    img = image.load_img(img_path, color_mode='grayscale', target_size=(224, 224))
    img_array = image.img_to_array(img)
    
    # Normalize the image to range [0, 1]
    img_array = img_array / 255.0
    
    # Reshape the image to match the model's input shape (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_class(img_array):
    # Predict the class of the image using the model
    prediction = model.predict(img_array)
    
    # Get the class label with the highest probability
    predicted_class_index = np.argmax(prediction)
    return shape_labels[predicted_class_index]

def evaluate_model(directory):
    true_labels = []
    predictions = []

    for label in shape_labels:
        try:
            label_dir = os.path.join(directory, label)
            img_names = os.listdir(label_dir)
            
            # Wrap img_names with tqdm to display progress
            for img_name in tqdm(img_names, desc=f"Processing {label}", unit="image"):
                img_path = os.path.join(label_dir, img_name)
                
                # Load and preprocess the image
                img_array = load_and_preprocess_image(img_path)
                
                # Get the true label and predicted class
                true_labels.append(label)
                predicted_class = predict_class(img_array)
                predictions.append(predicted_class)
        except Exception as e:
            print(f"An error occurred: {e}")
            pass
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f'Accuracy: {accuracy * 100:.2f}%')

# Example usage:
test_directory = './dataset_images'
# evaluate_model(test_directory)
