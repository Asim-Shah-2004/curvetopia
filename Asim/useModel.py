from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the Keras model


# Preprocess the input image
def preprocess_image(image_path, img_size):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Run inference using the Keras model
def run_inference(model, image_path, img_size):
    # Preprocess the image
    input_data = preprocess_image(image_path, img_size)

    # Run the inference
    output_data = model.predict(input_data)

    # Post-process the output to display the image
    output_image = output_data[0, :, :, 0] * 255.0  # De-normalize to [0, 255]
    output_image = output_image.astype(np.uint8)
    return output_image

# Main function
def main():
    image_path = r'C:\Users\Asim\OneDrive\Desktop\projects\curvetopia\testData\image.png'
    img_size = 128  # Ensure this matches your model's input size

    # Load the Keras model
    model = load_model(r'C:\Users\Asim\OneDrive\Desktop\projects\curvetopia\Asim\shape_smoothness_model.h5')

    # Run inference
    output_image = run_inference(model, image_path, img_size)

    # Save or display the result
    cv2.imwrite('output_image.png', output_image)
    cv2.imshow('Smooth Image', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
