import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define functions to read CSV and detect shapes
def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def is_line(curve, threshold=1e-2):
    if len(curve) < 2:
        return False
    x_coords, y_coords = curve[:, 0], curve[:, 1]
    line = np.polyfit(x_coords, y_coords, 1)
    fitted_line = np.polyval(line, x_coords)
    error = np.sum((fitted_line - y_coords) ** 2)
    return error < threshold

def is_circle_or_ellipse(curve, threshold=1e-2):
    if len(curve) < 5:
        return False, None
    ellipse = cv2.fitEllipse(curve.astype(np.float32))
    center, axes, angle = ellipse
    a, b = axes[0] / 2, axes[1] / 2
    distances = np.sqrt(((curve[:, 0] - center[0]) ** 2) / a ** 2 + ((curve[:, 1] - center[1]) ** 2) / b ** 2)
    error = np.std(distances)
    if error < threshold:
        if np.isclose(a, b, atol=threshold):
            return True, "circle"
        return True, "ellipse"
    return False, None

def is_rectangle_or_rounded_rectangle(curve, threshold=1e-2):
    if len(curve) < 4:
        return False, None
    rect = cv2.minAreaRect(curve.astype(np.float32))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    error = np.sum(np.min(np.linalg.norm(curve[:, np.newaxis] - box[np.newaxis, :], axis=2), axis=1))
    if error < threshold:
        return True, "rectangle"
    return False, None

def is_regular_polygon(curve, num_sides, threshold=1e-2):
    if len(curve) < num_sides:
        return False
    angle = 2 * np.pi / num_sides
    for i in range(num_sides):
        j = (i + 1) % num_sides
        edge_vector = curve[j] - curve[i]
        length = np.linalg.norm(edge_vector)
        next_edge_vector = curve[(j + 1) % num_sides] - curve[j]
        next_length = np.linalg.norm(next_edge_vector)
        if not np.isclose(length, next_length, atol=threshold):
            return False
        angle_diff = np.arccos(np.clip(np.dot(edge_vector, next_edge_vector) / (length * next_length), -1, 1))
        if not np.isclose(angle_diff, angle, atol=threshold):
            return False
    return True

def is_star_shape(curve, threshold=1e-2):
    hull = cv2.convexHull(curve.astype(np.float32))
    if len(hull) < 6:
        return False
    return True

# Define functions to regularize shapes
def regularize_line(curve):
    return np.array([curve[0], curve[-1]])

def regularize_circle_or_ellipse(curve, shape_type):
    ellipse = cv2.fitEllipse(curve.astype(np.float32))
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
    rect = cv2.minAreaRect(curve.astype(np.float32))
    box = cv2.boxPoints(rect)
    return np.int0(box)

# Define function to classify and plot curves
def classify_and_plot(curves):
    plt.figure(figsize=(10, 10))
    for curve in curves:
        curve = np.array(curve)  # Ensure curve is a numpy array
        if is_line(curve):
            print("Detected: Line")
            curve = regularize_line(curve)
        elif is_circle_or_ellipse(curve)[0]:
            shape_type = "circle" if is_circle_or_ellipse(curve)[1] == "circle" else "ellipse"
            print(f"Detected: {shape_type.capitalize()}")
            curve = regularize_circle_or_ellipse(curve, shape_type)
        elif is_rectangle_or_rounded_rectangle(curve)[0]:
            shape_type = "rectangle" if is_rectangle_or_rounded_rectangle(curve)[1] == "rectangle" else "rounded rectangle"
            print(f"Detected: {shape_type.capitalize()}")
            curve = regularize_rectangle_or_rounded_rectangle(curve, shape_type)
        elif is_regular_polygon(curve, num_sides=5):
            print("Detected: Regular Polygon")
        elif is_star_shape(curve):
            print("Detected: Star Shape")
        else:
            print("Detected: Irregular Shape")

        plt.plot(curve[:, 0], curve[:, 1])

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Load and classify curves from CSV
curves = read_csv('problems/frag0.csv')
classify_and_plot(curves)

