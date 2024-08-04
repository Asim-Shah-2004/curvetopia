import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import Canvas
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = (224, 224)
MODEL_PATH = './star_model.keras'

# Load the trained model
model = load_model(MODEL_PATH)


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw and Predict")
        self.canvas = Canvas(root, width=IMAGE_SIZE[0], height=IMAGE_SIZE[1], bg='white')
        self.canvas.pack()

        self.image = Image.new('L', IMAGE_SIZE, color=255)  # Create a white canvas
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.root.bind("<ButtonRelease-1>", self.predict)

    def paint(self, event):
        x, y = event.x, event.y
        self.draw.ellipse([x-5, y-5, x+5, y+5], fill='black')
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='black')

    def preprocess_image(self, img):
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Rescale pixel values to [0, 1]
        return img_array

    def predict(self, event):
        img_array = self.preprocess_image(self.image)
        prediction = model.predict(img_array)[0][0]
        prediction_percentage = prediction * 100

        plt.figure(figsize=(6, 6))
        plt.imshow(self.image, cmap='gray')
        plt.title(f"Predicted: {'Star' if prediction > 0.5 else 'Non-Star'}\nConfidence: {prediction_percentage:.2f}%")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
