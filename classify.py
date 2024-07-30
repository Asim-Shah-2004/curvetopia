import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

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

def classify_shapes(contours):
    shape_info = []
    for cnt in contours:
        shape = "unidentified"
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) == 2:
            shape = "line"
        elif len(approx) >= 5:
            shape = "circle/ellipse"
        elif len(approx) == 4:
            shape = "rectangle/rounded rectangle"
        elif len(approx) > 4:
            shape = "polygon"
        shape_info.append((shape, approx))
    return shape_info

def is_ellipse(approx):
    return len(approx) >= 5

def is_rectangle(approx):
    if len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        return 0.95 <= ar <= 1.05
    return False

def is_star_shape(cnt):
    if len(cnt) > 5:
        center, _ = cv2.minEnclosingCircle(cnt)
        center = tuple(map(int, center))
        angles = []
        for point in cnt:
            angle = math.atan2(point[0][1] - center[1], point[0][0] - center[0])
            angles.append(angle)
        angles.sort()
        angle_diffs = [j - i for i, j in zip(angles[:-1], angles[1:])]
        if len(set(angle_diffs)) == len(angle_diffs):
            return True
    return False

points = read_csv("problems/isolated.csv")

# Process each set of points
for idx, image_points in enumerate(points):
    # Prepare a blank image
    image = np.zeros((500, 500), dtype=np.uint8)
    
    # Draw points onto the image
    for point in image_points[0]:
        cv2.circle(image, (int(point[0]), int(point[1])), 1, 255, -1)
    
    # Detect edges
    edges = cv2.Canny(image, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Classify shapes
    shapes = classify_shapes(contours)
    
    # Print classifications
    print(f"Image {idx}:")
    for shape, approx in shapes:
        if shape == "circle/ellipse" and is_ellipse(approx):
            shape = "ellipse"
        elif shape == "rectangle/rounded rectangle":
            if is_rectangle(approx):
                shape = "rectangle"
            else:
                shape = "rounded rectangle"
        elif shape == "polygon" and is_star_shape(approx):
            shape = "star shape"
        
        print(f" - {shape}")
