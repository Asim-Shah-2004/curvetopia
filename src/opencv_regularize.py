import numpy as np
import cv2
import matplotlib.pyplot as plt
import svgwrite
import cairosvg
import os
from typing import List, Tuple
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from scipy.spatial import ConvexHull

# Shape labels
shape_labels = {
    1: "circle",
    2: "ellipse",
    3: "square",
    4: "rectangle",
    5: "polygon",
    6: "star"
}

# Constants
IMAGE_SIZE = (224, 224)
MODEL_PATH = '../models/star_model.keras'
temp_dir = '../temp'
os.makedirs(f'{temp_dir}', exist_ok=True)

# Load the trained model
model = load_model(MODEL_PATH)


def preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size, color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def is_star_with_model(image_path):
    img_array = preprocess_image(image_path, IMAGE_SIZE)
    prediction = model.predict(img_array)
    return prediction[0] > 0.7


def detect_predominant_shape(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    predominant_shape = None
    predominant_label = -1

    for contour in contours:
        # Calculate area of contour
        area = cv2.contourArea(contour)

        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if is_star_with_model(image_path):
            shape, label = "star", 6
        else:
            # Analyze the shape
            shape, label = analyze_shape(approx, contour)

        # Update predominant shape if current contour has larger area
        if area > max_area:
            max_area = area
            predominant_shape = shape
            predominant_label = label

    # Display the image with only the predominant shape detected
    if predominant_shape:
        cv2.putText(image, shape_labels[predominant_label], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Predominant Shape", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return predominant_shape, predominant_label


def analyze_shape(approx, contour):
    # Analyze the shape based on the number of points
    if len(approx) == 3:
        return "polygon", 5  # Triangle is a polygon
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05 and is_square(approx):
            return "square", 3
        else:
            return "rectangle", 4
    elif len(approx) == 5:
        return "polygon", 5  # Pentagon is a polygon
    elif len(approx) == 6:
        return "polygon", 5  # Hexagon is a polygon
    elif len(approx) > 6:
        if is_circle(contour):
            return "circle", 1
        elif is_ellipse(contour):
            return "ellipse", 2
        else:
            return "polygon", 5
    else:
        return "unknown", -1


def is_square(approx):
    if len(approx) != 4:
        return False
    (x, y, w, h) = cv2.boundingRect(approx)
    aspect_ratio = w / float(h)
    if not 0.95 <= aspect_ratio <= 1.05:
        return False
    angles = []
    for i in range(4):
        p1 = approx[i % 4][0]
        p2 = approx[(i + 1) % 4][0]
        p3 = approx[(i + 2) % 4][0]
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(cosine_angle))
        angles.append(angle)
    return all(85 <= angle <= 95 for angle in angles)


def is_circle(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return 0.8 <= circularity <= 1.2


def is_ellipse(contour):
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        _, (MA, ma), _ = ellipse
        if MA / ma < 1.2:
            return True
    return False


# def is_star(approx, contour):
#     if len(approx) == 10:
#         angles = []
#         for i in range(10):
#             pt1 = approx[i % 10][0]
#             pt2 = approx[(i + 1) % 10][0]
#             pt3 = approx[(i + 2) % 10][0]
#             v1 = np.array([pt1[0] - pt2[0], pt1[1] - pt2[1]])
#             v2 = np.array([pt3[0] - pt2[0], pt3[1] - pt2[1]])
#             cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#             angle = np.arccos(cosine_angle)
#             angles.append(np.degrees(angle))
#         avg_angle = np.mean(angles)
#         if 144 <= avg_angle <= 216:
#             return True
#     return False


def read_csv(csv_path: str) -> List[List[np.ndarray]]:
    try:
        # Load CSV data
        data = np.genfromtxt(csv_path, delimiter=',', dtype=float)

        # Extract unique IDs
        ids = np.unique(data[:, 0])

        # Group points by ID
        path_XYs = []
        for id in ids:
            # Filter rows for this ID
            curve_data = data[data[:, 0] == id][:, 1:]

            # Extract unique curve IDs
            curve_ids = np.unique(curve_data[:, 0])
            XYs = []

            for j in curve_ids:
                XY = curve_data[curve_data[:, 0] == j][:, 1:]
                XYs.append(XY)

            path_XYs.append(XYs)
        return path_XYs  # [[[X1, Y!], [X2, Y2], ...], [[X1, Y1], [X2, Y2], ...], ...]
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []


def remove_whitespace(image_path: str) -> None:
    image = Image.open(image_path).convert('RGBA')
    gray = image.convert('L')
    bbox = gray.getbbox()
    if bbox:
        cropped_image = image.crop(bbox)
        cropped_image.save(image_path)
    else:
        image.save(image_path)


def polylines2svg(XY: np.ndarray, svg_path: str, colour: str = 'blue', stroke_width: int = 2) -> Tuple[str, str]:
    if XY.ndim != 2 or XY.shape[1] != 2:
        raise ValueError("XY should be a 2D array with shape (rows, 2)")

    # Determine the width and height of the SVG
    min_x, min_y = np.min(XY, axis=0)
    max_x, max_y = np.max(XY, axis=0)
    W = max_x - min_x + 10
    H = max_y - min_y + 10

    # Create a new SVG drawing
    dwg = svgwrite.Drawing(svg_path, profile='tiny', size=(W, H))
    dwg.viewbox(0, 0, W, H)
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill='white'))

    # Adjust path data to account for padding
    path_data = [("M", (XY[0, 0] - min_x + 5, XY[0, 1] - min_y + 5))]  # Add padding to path
    for i in range(1, len(XY)):
        path_data.append(("L", (XY[i, 0] - min_x + 5, XY[i, 1] - min_y + 5)))
    # path_data.append(("Z", None))  # Uncomment if you need to close the path

    dwg.add(dwg.path(d=path_data, fill='none', stroke=colour, stroke_width=stroke_width))
    dwg.save()

    # Convert SVG to PNG
    png_path = svg_path.replace('.svg', '.png')
    cairosvg.svg2png(url=svg_path, write_to=png_path, parent_width=W, parent_height=H,
                     output_width=W, output_height=H, background_color='white')

    # Remove whitespace from the PNG
    remove_whitespace(png_path)

    return svg_path, png_path


def regularize_line(curve):
    """
    Regularizes a line shape to be closed.
    """
    if len(curve) < 2:
        return curve
    return np.array([curve[0], curve[-1], curve[0]])  # Adding the starting point to close the line


def regularize_circle_or_ellipse(curve, shape_type):
    """
    Regularizes circle or ellipse shapes to be closed.
    """
    if len(curve) < 5:
        return np.vstack((curve, curve[0]))  # Not enough points to fit ellipse, close it as is

    # Extract coordinates
    points = curve[:, :2].astype(np.float32)

    # Fit an ellipse
    ellipse = cv2.fitEllipse(points)
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

    # Close the shape by adding the starting point
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    return np.vstack((x, y)).T


def regularize_rectangle_or_rounded_rectangle(curve, shape_type):
    """
    Regularizes rectangle or rounded rectangle shapes to be closed.
    """
    if len(curve) < 4:
        return np.vstack((curve, curve[0]))  # Not enough points to fit rectangle, close it as is

    # Extract coordinates
    points = curve[:, :2].astype(np.float32)

    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    if shape_type == "rectangle":
        return np.vstack((box, box[0]))  # Adding the starting point to close the rectangle
    elif shape_type == "rounded_rectangle":
        # Example: Approximate rounded corners (dummy implementation)
        return np.vstack((box, box[0]))  # Just to illustrate; refine this as needed

    return np.vstack((box, box[0]))  # Ensure the shape is closed


def regularize_star(curve: np.ndarray, num_points: int = 5) -> np.ndarray:
    if len(curve) < 5:
        # Not enough points to form a star, return the original shape with closure
        return np.vstack((curve, curve[0]))

    # Extract coordinates
    points = curve[:, :2]

    # Calculate the centroid of the points
    center = np.mean(points, axis=0)

    # Calculate distances and angles from the center to each point
    distances = np.linalg.norm(points - center, axis=1)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

    # Sort angles and corresponding distances
    sorted_indices = np.argsort(angles)
    sorted_angles = angles[sorted_indices]
    sorted_distances = distances[sorted_indices]

    # Generate the star shape
    star_angles = np.linspace(0, 2 * np.pi, num_points * 2, endpoint=False)
    star_radii = np.zeros(num_points * 2)
    
    # Alternating radius for star points and inner points
    outer_radius = np.max(sorted_distances)
    inner_radius = np.min(sorted_distances)
    star_radii[::2] = outer_radius  # Outer points
    star_radii[1::2] = inner_radius  # Inner points

    # Interpolate the original shape onto the star shape
    interp_distances = np.interp(star_angles, sorted_angles, sorted_distances, period=2*np.pi)
    
    # Blend the original shape with the ideal star shape
    blend_factor = 0.5  # Adjust this value to control how much of the original shape is preserved
    blended_radii = blend_factor * interp_distances + (1 - blend_factor) * star_radii

    # Generate x and y coordinates
    x = center[0] + blended_radii * np.cos(star_angles)
    y = center[1] + blended_radii * np.sin(star_angles)

    # Create the star shape
    star_shape = np.column_stack((x, y))

    # Close the shape by adding the first point at the end
    closed_star_shape = np.vstack((star_shape, star_shape[0]))

    # Return the closed star shape
    return closed_star_shape

def classify_and_plot(path_XYs: List[List[np.ndarray]]) -> None:
    colours = ['blue', 'red', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'gray']

    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(path_XYs):
        c = colours[i % len(colours)]
        for j, XY in enumerate(XYs):
            XY = np.array(XY)

            svg_path, png_path = polylines2svg(XY, f"{temp_dir}/curve_{i}_{j}.svg", colour='blue')
            shape, label = detect_predominant_shape(f"{temp_dir}/curve_{i}_{j}.png")
            print(f"Predominant Shape: {shape}, Label: {label}")

            # Regularize shape based on detected shape
            if shape == 'line':
                regularized_XY = regularize_line(XY)
            elif shape in ['circle', 'ellipse']:
                regularized_XY = regularize_circle_or_ellipse(XY, shape)
            elif shape in ['rectangle', 'rounded_rectangle']:
                regularized_XY = regularize_rectangle_or_rounded_rectangle(XY, shape)
            elif shape == 'star':
                regularized_XY = regularize_star(XY)
            else:
                regularized_XY = XY  # No regularization for unknown shapes

            # Plotting the regularized shape
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
            ax.plot(regularized_XY[:, 0], regularized_XY[:, 1], c=c, linewidth=2)

    ax.set_aspect('equal')
    plt.show()


# Example usage
csv_path = '../problems/isolated.csv'
path_XYs = read_csv(csv_path)
classify_and_plot(path_XYs)
