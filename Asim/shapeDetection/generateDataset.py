import cv2
import numpy as np
import os
import random

# Base directory for the dataset
output_dir = 'dataset_images/'

# Function to create the directory if it doesn't exist
def ensure_dir_exists(directory):
    os.makedirs(directory, exist_ok=True)

# Function to draw a hand-drawn circle
def draw_hand_drawn_circle(image, center, radius, color):
    num_points = 20 
    points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        random_offset = random.randint(-5, 5)
        x = int(center[0] + (radius + random_offset) * np.cos(angle))
        y = int(center[1] + (radius + random_offset) * np.sin(angle))
        points.append((x, y))
    points = np.array(points, np.int32).reshape((-1, 1, 2))
    for _ in range(5):
        offset_x = random.randint(-3, 3)
        offset_y = random.randint(-3, 3)
        offset_points = points + [offset_x, offset_y]
        cv2.polylines(image, [offset_points], isClosed=True, color=color, thickness=random.randint(1, 3), lineType=cv2.LINE_AA)

# Function to draw a hand-drawn line
def draw_hand_drawn_line(image, start, end, color):
    num_segments = 5 
    max_offset = 2     
    points = [start]
    for i in range(1, num_segments):
        t = i / num_segments
        intermediate_point = (
            int(start[0] * (1 - t) + end[0] * t + random.randint(-max_offset, max_offset)),
            int(start[1] * (1 - t) + end[1] * t + random.randint(-max_offset, max_offset))
        )
        points.append(intermediate_point)
    points.append(end)
    for i in range(len(points) - 1):
        cv2.line(image, points[i], points[i + 1], color, thickness=random.randint(1, 3), lineType=cv2.LINE_AA)

# Function to draw a hand-drawn rectangle
def draw_hand_drawn_rectangle(image, top_left, width, height, color):
    tl_x_variability = random.randint(-10, 10)
    tl_y_variability = random.randint(-10, 10)
    tr_x_variability = random.randint(-10, 10)
    tr_y_variability = random.randint(-10, 10)
    bl_x_variability = random.randint(-10, 10)
    bl_y_variability = random.randint(-10, 10)
    br_x_variability = random.randint(-10, 10)
    br_y_variability = random.randint(-10, 10)

    top_left = (top_left[0] + tl_x_variability, top_left[1] + tl_y_variability)
    top_right = (top_left[0] + width + tr_x_variability, top_left[1] + tr_y_variability)
    bottom_left = (top_left[0] + bl_x_variability, top_left[1] + height + bl_y_variability)
    bottom_right = (top_left[0] + width + br_x_variability, top_left[1] + height + br_y_variability)

    draw_hand_drawn_line(image, top_left, top_right, color)
    draw_hand_drawn_line(image, top_right, bottom_right, color)
    draw_hand_drawn_line(image, bottom_right, bottom_left, color)
    draw_hand_drawn_line(image, bottom_left, top_left, color)
    
# Function to draw a hand-drawn star
def draw_hand_drawn_star(image, center, size, color):
    points = []
    for i in range(10):
        angle = i * np.pi / 5
        length = size if i % 2 == 0 else size / 2
        length += random.randint(-10, 10)
        x = int(center[0] + length * np.cos(angle))
        y = int(center[1] - length * np.sin(angle))
        points.append((x, y))
    for j in range(10):
        start = points[j]
        end = points[(j + 1) % 10]
        draw_hand_drawn_line(image, start, end, color)

# Function to draw a hand-drawn ellipse
def draw_hand_drawn_ellipse(image, center, axes, angle, color):
    num_points = 10
    points = []
    for i in range(num_points):
        theta = 2 * np.pi * i / num_points
        random_offset = random.randint(-5, 5)
        x = int(center[0] + (axes[0] + random_offset) * np.cos(theta) * np.cos(angle) - (axes[1] + random_offset) * np.sin(theta) * np.sin(angle))
        y = int(center[1] + (axes[0] + random_offset) * np.cos(theta) * np.sin(angle) + (axes[1] + random_offset) * np.sin(theta) * np.cos(angle))
        points.append((x, y))
    points = np.array(points, np.int32).reshape((-1, 1, 2))
    for _ in range(5):
        offset_x = random.randint(-3, 3)
        offset_y = random.randint(-3, 3)
        offset_points = points + [offset_x, offset_y]
        cv2.polylines(image, [offset_points], isClosed=True, color=color, thickness=random.randint(1, 3), lineType=cv2.LINE_AA)
        
# Function to draw a hand-drawn polygon
def draw_hand_drawn_polygon(image, center, sides, radius, color):
    points = []
    for i in range(sides):
        angle = i * 2 * np.pi / sides
        variable_radius = radius + random.randint(-10, 10)
        x = int(center[0] + variable_radius * np.cos(angle))
        y = int(center[1] - variable_radius * np.sin(angle))
        points.append((x, y))
    for i in range(sides):
        start = points[i]
        end = points[(i + 1) % sides]
        draw_hand_drawn_line(image, start, end, color)

def create_hand_drawn_image(shape_type, file_path):
    img = np.ones((256, 256, 3), dtype=np.uint8) * 255  
    color = (0, 0, 0)

    shape_params = {
        'circle': lambda: draw_hand_drawn_circle(img, (128, 128), 100, color),
        'rectangle': lambda: draw_hand_drawn_rectangle(img, (50, 50), 150, 100, color),
        'star': lambda: draw_hand_drawn_star(img, (128, 128), 100, color),
        'square': lambda: draw_hand_drawn_rectangle(img, (50, 50), 100, 100, color),
        'hexagon': lambda: draw_hand_drawn_polygon(img, (128, 128), 6, 100, color),
        'polygon': lambda: draw_hand_drawn_polygon(img, (128, 128), 8, 100, color),
        'ellipse': lambda: draw_hand_drawn_ellipse(img, (128, 128), (100, 50), 0, color),
        'line': lambda: draw_hand_drawn_line(img, (50, 128), (200, 128), color)
    }

    if shape_type in shape_params:
        shape_params[shape_type]()

    img = cv2.GaussianBlur(img, (5, 5), 0)

    cv2.imwrite(file_path, img)

# Function to create dataset
def create_dataset(num_images_per_class):
    shapes = ['circle', 'rectangle', 'star', 'square', 'hexagon', 'polygon', 'ellipse', 'line']
    split_ratio = 0.8
    num_train = int(num_images_per_class * split_ratio)
    num_test = num_images_per_class - num_train

    for shape in shapes:
        shape_train_dir = os.path.join(output_dir, 'train', shape)
        shape_test_dir = os.path.join(output_dir, 'test', shape)
        ensure_dir_exists(shape_train_dir)
        ensure_dir_exists(shape_test_dir)
        
        for i in range(num_train):
            file_path = os.path.join(shape_train_dir, f'{shape}_{i}.png')
            create_hand_drawn_image(shape, file_path)

        for i in range(num_test):
            file_path = os.path.join(shape_test_dir, f'{shape}_{i}.png')
            create_hand_drawn_image(shape, file_path)

# Number of images per class
num_images_per_class = 200

# Create the dataset
create_dataset(num_images_per_class)

print('Image generation complete.')

