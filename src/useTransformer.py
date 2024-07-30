import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Load the trained model and feature extractor
model = ViTForImageClassification.from_pretrained('./results')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

# Define the image path
image_path = r'C:\Users\Asim\Downloads\test.png'  # Replace with the path to your image

# Define class labels
class_labels = ['circle', 'square', 'triangle', 'rectangle', 'pentagon', 'hexagon', 'octagon', 'star']  # Replace with your actual class labels

# Load and preprocess the image
image = Image.open(image_path)

# Convert to RGB and resize
if image.mode != 'RGB':
    image = image.convert('RGB')
image = image.resize((224, 224))  # Resize to 224x224 if needed

inputs = feature_extractor(images=image, return_tensors="pt")

# Make prediction
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

# Map the predicted index to class label
predicted_class_label = class_labels[predicted_class_idx]

print(f'Predicted class index: {predicted_class_idx}')
print(f'Predicted class label: {predicted_class_label}')
