import cv2
import numpy as np

# Shape labels
shape_labels = {
    0: "circle",
    1: "ellipse",
    2: "hexagon",
    3: "line",
    4: "polygon",
    5: "rectangle",
    6: "square",
    7: "star"
}

def detect_predominant_shape(image_path):
    # Load the image
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
        
        # Analyze the shape
        if len(approx) == 3:
            shape = "triangle"
            label = 4  # Assume polygon label for triangles
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.95 <= aspect_ratio <= 1.05 and is_square(approx):
                shape = "square"
                label = 6
            else:
                shape = "rectangle"
                label = 5
        elif len(approx) == 5:
            shape = "pentagon"
            label = 4
        elif len(approx) == 6:
            shape = "hexagon"
            label = 2
        elif len(approx) > 6:
            if is_circle(contour):
                shape = "circle"
                label = 0
            elif is_ellipse(contour):
                shape = "ellipse"
                label = 1
            elif is_star(approx, contour):
                shape = "star"
                label = 7
            else:
                shape = "polygon"
                label = 4
        else:
            continue
        
        # Update predominant shape if current contour has larger area
        if area > max_area:
            max_area = area
            predominant_shape = shape
            predominant_label = label
        
    # Display the image with only the predominant shape detected
    if predominant_shape:
        cv2.putText(image, predominant_shape, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Predominant Shape", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return predominant_shape, predominant_label

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
    if all(85 <= angle <= 95 for angle in angles):
        return True
    return False

def is_star(approx, contour):
    if len(approx) == 10:
        angles = []
        for i in range(10):
            pt1 = approx[i % 10][0]
            pt2 = approx[(i + 1) % 10][0]
            pt3 = approx[(i + 2) % 10][0]
            v1 = np.array([pt1[0] - pt2[0], pt1[1] - pt2[1]])
            v2 = np.array([pt3[0] - pt2[0], pt3[1] - pt2[1]])
            cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(cosine_angle)
            angles.append(np.degrees(angle))
        avg_angle = np.mean(angles)
        if 144 <= avg_angle <= 216:
            return True
    return False

# Example usage
image_path = r'C:\Users\Asim\OneDrive\Desktop\projects\curvetopia\testData\image.png'
shape, label = detect_predominant_shape(image_path)
print(f"Predominant Shape: {shape}, Label: {label}")
