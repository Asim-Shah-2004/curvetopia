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


def regularize_shape(curve: np.ndarray, shape_type: str) -> np.ndarray:
    """
    Regularizes a shape based on its type.

    Args:
        curve (np.ndarray): The input curve points.
        shape_type (str): The type of shape to regularize.

    Returns:
        np.ndarray: The regularized shape points.
    """
    regularization_functions = {
        "line": regularize_line,
        "circle": lambda c: regularize_ellipse(c, is_circle=True),
        "ellipse": lambda c: regularize_ellipse(c, is_circle=False),
        "rectangle": lambda c: regularize_rectangle(c, rounded=False),
        "rounded_rectangle": lambda c: regularize_rectangle(c, rounded=True),
        "star": regularize_star
    }

    regularize_func = regularization_functions.get(shape_type.lower())
    if regularize_func is None:
        print(f"Unsupported shape type: {shape_type}")
        return curve

    return regularize_func(curve)


def regularize_line(curve: np.ndarray) -> np.ndarray:
    """Regularizes a line shape to be closed."""
    if len(curve) < 2:
        return curve
    return np.vstack((curve[0], curve[-1], curve[0]))


def regularize_ellipse(curve: np.ndarray, is_circle: bool = False) -> np.ndarray:
    """
    Regularizes circle or ellipse shapes to be closed.

    Args:
        curve (np.ndarray): The input curve points.
        is_circle (bool): Whether to treat the shape as a circle.

    Returns:
        np.ndarray: The regularized shape points.
    """
    if len(curve) < 5:
        return np.vstack((curve, curve[0]))

    points = curve[:, :2].astype(np.float32)
    ellipse = cv2.fitEllipse(points)
    center, axes, angle = ellipse

    if is_circle:
        radius = np.mean(axes) / 2
        t = np.linspace(0, 2 * np.pi, 100)
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
    else:
        a, b = axes[0] / 2, axes[1] / 2
        t = np.linspace(0, 2 * np.pi, 100)
        angle_rad = np.radians(angle)
        x = center[0] + a * np.cos(t) * np.cos(angle_rad) - b * np.sin(t) * np.sin(angle_rad)
        y = center[1] + a * np.cos(t) * np.sin(angle_rad) + b * np.sin(t) * np.cos(angle_rad)

    return np.column_stack((np.append(x, x[0]), np.append(y, y[0])))


def regularize_rectangle(curve: np.ndarray, rounded: bool = False) -> np.ndarray:
    """
    Regularizes rectangle or rounded rectangle shapes to be closed.

    Args:
        curve (np.ndarray): The input curve points.
        rounded (bool): Whether to create a rounded rectangle.

    Returns:
        np.ndarray: The regularized shape points.
    """
    if len(curve) < 4:
        return np.vstack((curve, curve[0]))

    points = curve[:, :2].astype(np.float32)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)

    if not rounded:
        return np.vstack((box, box[0]))

    # Implement rounded corners
    corner_radius = min(rect[1]) * 0.1  # 10% of the smaller side as corner radius
    return create_rounded_rectangle(box, corner_radius)


def create_rounded_rectangle(rect: np.ndarray, radius: float) -> np.ndarray:
    """Creates a rounded rectangle from a given rectangle and corner radius."""
    corners = [rect[i] for i in range(4)]
    rounded_rect = []

    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]

        # Calculate the direction vector
        dir_vec = p2 - p1
        dir_vec_norm = dir_vec / np.linalg.norm(dir_vec)

        # Calculate start and end points of the rounded corner
        start = p1 + dir_vec_norm * radius
        end = p2 - dir_vec_norm * radius

        # Add the straight line
        rounded_rect.extend([start, end])

        # Add the rounded corner
        center = p2 - dir_vec_norm * radius
        next_dir = corners[(i + 2) % 4] - p2
        next_dir_norm = next_dir / np.linalg.norm(next_dir)
        angle_start = np.arctan2(dir_vec_norm[1], dir_vec_norm[0])
        angle_end = np.arctan2(next_dir_norm[1], next_dir_norm[0])

        if angle_end < angle_start:
            angle_end += 2 * np.pi

        angles = np.linspace(angle_start, angle_end, 10)
        for angle in angles:
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            rounded_rect.append([x, y])

    return np.array(rounded_rect)


def regularize_star(curve: np.ndarray, num_points: int = 5) -> np.ndarray:
    """
    Regularizes a star shape.

    Args:
        curve (np.ndarray): The input curve points.
        num_points (int): The number of points in the star.

    Returns:
        np.ndarray: The regularized star shape points.
    """
    if len(curve) < 5:
        return np.vstack((curve, curve[0]))

    points = curve[:, :2]
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

    sorted_indices = np.argsort(angles)
    sorted_angles = angles[sorted_indices]
    sorted_distances = distances[sorted_indices]

    star_angles = np.linspace(0, 2 * np.pi, num_points * 2, endpoint=False)
    outer_radius = np.max(sorted_distances)
    inner_radius = np.min(sorted_distances)
    star_radii = np.tile([outer_radius, inner_radius], num_points)

    interp_distances = np.interp(star_angles, sorted_angles, sorted_distances, period=2*np.pi)
    blend_factor = 0.5
    blended_radii = blend_factor * interp_distances + (1 - blend_factor) * star_radii

    x = center[0] + blended_radii * np.cos(star_angles)
    y = center[1] + blended_radii * np.sin(star_angles)

    return np.column_stack((np.append(x, x[0]), np.append(y, y[0])))


def classify_and_plot(path_XYs: List[List[np.ndarray]]) -> None:
    colours = ['blue', 'red', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'gray']

    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    regularized_path_XYs = []
    for i, XYs in enumerate(path_XYs):
        c = colours[i % len(colours)]
        regularized_XYs = []
        for j, XY in enumerate(XYs):
            XY = np.array(XY)

            svg_path, png_path = polylines2svg(XY, f"{temp_dir}/curve_{i}_{j}.svg", colour='blue')
            shape, label = detect_predominant_shape(f"{temp_dir}/curve_{i}_{j}.png")
            print(f"Predominant Shape: {shape}, Label: {label}")

            regularized_XY = regularize_shape(XY, shape)
            regularized_XYs.append(regularized_XY)

            # Plotting the regularized shape
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
            ax.plot(regularized_XY[:, 0], regularized_XY[:, 1], c=c, linewidth=2)
        
        regularized_path_XYs.append(regularized_XYs)

    ax.set_aspect('equal')
    plt.show()
    save_to_csv(regularized_path_XYs, output_csv_path)

def save_to_csv(path_XYs: List[List[np.ndarray]], output_csv_path: str) -> None:
    """
    Saves the path_XYs data to a CSV file.

    Args:
        path_XYs (List[List[np.ndarray]]): The path XY data to save.
        output_csv_path (str): The path to the output CSV file.
    """
    with open(output_csv_path, 'w') as file:
        for i, XYs in enumerate(path_XYs):
            for j, XY in enumerate(XYs):
                for point in XY:
                    file.write(f"{i},{j},{point[0]},{point[1]}\n")


# Example usage
csv_path = '../problems/isolated.csv'
output_csv_path = '../problems/regularized_shapes.csv'
path_XYs = read_csv(csv_path)
classify_and_plot(path_XYs)
