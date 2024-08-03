import numpy as np
import cv2
import matplotlib.pyplot as plt
import svgwrite
import cairosvg
from keras.models import load_model
from PIL import Image
import os
from scipy.spatial import ConvexHull

# Define a color map for different shapes
color_map = {
    "Line": "b",
    "Circle": "r",
    "Ellipse": "g",
    "Rectangle": "c",
    "Rounded rectangle": "m",
    "Pentagon": "y",
    "Star Shape": "orange",
    "Irregular Shape": "gray"
}

# Load your Keras model
model = load_model('../models/shapes_model.keras')


def read_csv(csv_path):
    """
    Reads a CSV file and processes it into a list of paths with coordinates.
    """
    try:
        np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
        path_XYs = []
        for i in np.unique(np_path_XYs[:, 0]):
            npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
            XYs = []
            for j in np.unique(npXYs[:, 0]):
                XY = np_path_XYs[np_path_XYs[:, 0] == j][:, 1:]
                XYs.append(XY)
            path_XYs.append(XYs)
        return path_XYs
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []


def polylines2svg(paths_XYs, svg_path, colours=['blue', 'red', 'green', 'yellow']):
    """
    Converts a list of paths to an SVG file and then to a PNG file.
    """
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            if XY.ndim == 2 and XY.shape[1] == 3:  # Ensure XY is a 2D array with shape (n, 3)
                W, H = max(W, np.max(XY[:, 1])), max(H, np.max(XY[:, 2]))

    if W == 0 or H == 0:
        print(f"Warning: Width or Height is zero. W: {W}, H: {H}")
        W, H = 1, 1  # Avoid zero dimensions by setting to minimum values

    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)

    # Create a new SVG drawing
    dwg = svgwrite.Drawing(svg_path, profile='tiny', size=(W, H))
    dwg.viewbox(0, 0, W, H)
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill='white'))

    group = dwg.g()
    for i, path in enumerate(paths_XYs):
        path_data = []
        c = colours[i % len(colours)]
        for XY in path:
            if XY.ndim == 2 and XY.shape[1] == 3:
                path_data.append(("M", (XY[0, 1], XY[0, 2])))
                for j in range(1, len(XY)):
                    path_data.append(("L", (XY[j, 1], XY[j, 2])))
                if not np.allclose(XY[0, 1:], XY[-1, 1:]):
                    path_data.append(("Z", None))
        if path_data:
            group.add(dwg.path(d=path_data, fill=c, stroke='none', stroke_width=2))

    dwg.add(group)
    dwg.save()

    png_path = svg_path.replace('.svg', '.png')
    fact = max(1, 1024 // min(H, W))
    cairosvg.svg2png(url=svg_path, write_to=png_path, parent_width=W, parent_height=H,
                     output_width=fact*W, output_height=fact*H, background_color='white')

    return svg_path, png_path


def curve_to_png(curve, file_path):
    """
    Converts a curve to a PNG image file.
    """
    try:
        print(f"Converting curve with {len(curve)} points to PNG.")
        svg_path, png_path = polylines2svg([curve], file_path.replace('.png', '.svg'))
        return png_path
    except Exception as e:
        print(f"Error converting curve to PNG: {e}")
        return None


def preprocess_image(png_path):
    """
    Preprocesses the image for model input.
    """
    try:
        img = Image.open(png_path).convert('L')  # Convert to grayscale
        img = img.resize((224, 224))  # Resize to 224x224
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def predict_shape(img_array):
    """
    Predicts the shape type using the model.
    """
    try:
        prediction = model.predict(img_array)
        shape_types = ['line', 'triangle', 'square', 'circle', 'rectangle', 'star', 'regular_polygon']
        return shape_types[np.argmax(prediction)]
    except Exception as e:
        print(f"Error predicting shape: {e}")
        return "Unknown"


def regularize_line(curve):
    """
    Regularizes a line shape.
    """
    return np.array([curve[0], curve[-1]])


def regularize_circle_or_ellipse(curve, shape_type):
    """
    Regularizes circle or ellipse shapes.
    """
    ellipse = cv2.fitEllipse(curve[:, 1:].astype(np.float32))
    center, axes, angle = ellipse
    if shape_type == "circle":
        radius = (axes[0] + axes[1]) / 4
        t = np.linspace(0, 2 * np.pi, 100)
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
    else:
        a, b = axes[0] / 2, axes[1] / 2
        t = np.linspace(0, 2 * np.pi, 100)
        x = center[0] + a * np.cos(t) * np.cos(np.radians(angle)) - b * np.sin(t) * np.sin(np.radians(angle))
        y = center[1] + a * np.cos(t) * np.sin(np.radians(angle)) + b * np.sin(t) * np.cos(np.radians(angle))
    return np.vstack((x, y)).T


def regularize_rectangle_or_rounded_rectangle(curve, shape_type):
    """
    Regularizes rectangle or rounded rectangle shapes.
    """
    rect = cv2.minAreaRect(curve[:, 1:].astype(np.float32))
    box = cv2.boxPoints(rect)
    return np.int32(box)


def regularize_star(curve):
    """
    Regularizes a star shape by approximating it with its convex hull.
    """
    points = curve[:, 1:].astype(np.float32)
    if len(points) < 5:
        return points  # Not enough points to form a star
    hull = ConvexHull(points)
    return points[hull.vertices]


def classify_and_plot(path_XYs):
    """
    Classifies shapes, regularizes them, and plots the results.
    """
    i = 0
    all_curves = []

    # Create a temporary directory for PNG files
    temp_dir = 'temp'
    os.makedirs(temp_dir, exist_ok=True)

    for shape in path_XYs:
        for curve in shape:
            curve = np.array(curve)  # Ensure curve is a numpy array

            # Convert curve to PNG and classify
            png_path = curve_to_png(curve, os.path.join(temp_dir, f"temp_{i}.png"))
            if png_path is None:
                continue  # Skip if PNG conversion failed
            img_array = preprocess_image(png_path)
            if img_array is None:
                continue  # Skip if image preprocessing failed
            shape_type = predict_shape(img_array)

            # Regularize shape based on classification
            if shape_type == "line":
                curve = regularize_line(curve)
            elif shape_type == "rectangle" or shape_type == "square":
                curve = regularize_rectangle_or_rounded_rectangle(curve, shape_type)
            elif shape_type == "circle":
                curve = regularize_circle_or_ellipse(curve, shape_type)
            elif shape_type == "ellipse":
                curve = regularize_circle_or_ellipse(curve, shape_type)
            elif shape_type == "star":
                curve = regularize_star(curve)
            elif shape_type == "regular_polygon":
                curve = regularize_star(curve)
            else:
                curve = np.array(curve)  # Default case for unknown shapes

            all_curves.append(curve)

            # Plot the curves
            plt.figure(figsize=(8, 8))
            for curve in all_curves:
                plt.plot(curve[:, 0], curve[:, 1], 'o-')
            plt.title(f"Shape Classification: {shape_type}")
            plt.show()

    # Save SVG and PNG files
    for i, curve in enumerate(all_curves):
        svg_path, png_path = polylines2svg([curve], f'{temp_dir}/shape_{i}.svg')
        print(f"SVG and PNG files saved: {svg_path}, {png_path}")

    # Clean up temporary directory
    for file_name in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file_name))
    os.rmdir(temp_dir)


# Example usage
path_XYs = read_csv('../problems/isolated.csv')
classify_and_plot(path_XYs)
