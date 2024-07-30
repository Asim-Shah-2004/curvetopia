import numpy as np
import cv2
import os

def rotate_image(image, rotation_angle):
    image_center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    return rotated_image

def is_within_bounds(coords, width, height, margin=0):
    return all(0 + margin <= x < width - margin and 0 + margin <= y < height - margin for x, y in coords)

def add_path_noise(points, noise_level=1):  # Reduced noise level
    noisy_points = []
    for (x, y) in points:
        noisy_x = x + np.random.uniform(-noise_level, noise_level)
        noisy_y = y + np.random.uniform(-noise_level, noise_level)
        noisy_points.append((int(noisy_x), int(noisy_y)))
    return noisy_points

def draw_wobbly_line(img, pt1, pt2, color, thickness, noise_level):
    num_points = 20  # Number of intermediate points in the wobbly line
    line_points = np.linspace(pt1, pt2, num_points)
    wobbly_line_points = add_path_noise(line_points, noise_level)
    for i in range(len(wobbly_line_points) - 1):
        cv2.line(img, tuple(wobbly_line_points[i]), tuple(wobbly_line_points[i + 1]), color, thickness)

def draw_wobbly_circle(img, center, radius, color, thickness, noise_level):
    num_points = 100  # Number of points to draw the wobbly circle
    circle_points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        circle_points.append((x, y))
    wobbly_circle_points = add_path_noise(circle_points, noise_level)
    cv2.polylines(img, [np.array(wobbly_circle_points, np.int32)], isClosed=True, color=color, thickness=thickness)

def draw_wobbly_ellipse(img, center, axes, angle, color, thickness, noise_level):
    num_points = 100  # Number of points to draw the wobbly ellipse
    ellipse_points = []
    for i in range(num_points):
        theta = 2 * np.pi * i / num_points
        x = center[0] + axes[0] * np.cos(theta) * np.cos(np.radians(angle)) - axes[1] * np.sin(theta) * np.sin(np.radians(angle))
        y = center[1] + axes[0] * np.cos(theta) * np.sin(np.radians(angle)) + axes[1] * np.sin(theta) * np.cos(np.radians(angle))
        ellipse_points.append((x, y))
    wobbly_ellipse_points = add_path_noise(ellipse_points, noise_level)
    cv2.polylines(img, [np.array(wobbly_ellipse_points, np.int32)], isClosed=True, color=color, thickness=thickness)

def get_bounding_box(coords):
    x_coords, y_coords = zip(*coords)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return (x_min, y_min, x_max, y_max)

def convert_bbox_to_yolo_format(x_min, y_min, x_max, y_max, img_width, img_height):
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return (x_center, y_center, width, height)

def save_annotation_file(file_path, class_id, bbox, img_width, img_height):
    with open(file_path, 'w') as f:
        x_center, y_center, width, height = bbox
        f.write(f'{class_id} {x_center} {y_center} {width} {height}\n')

def generate_geometric_shapes_dataset(num_samples_per_class=50, output_directory='../dataset2'):
    shape_categories = ['line', 'triangle', 'square', 'circle', 'ellipse', 'rectangle', 'star', 'regular_polygon']
    image_size = 224
    margin = 10

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for shape_id, shape in enumerate(shape_categories):
        shape_directory = os.path.join(output_directory, shape)
        if not os.path.exists(shape_directory):
            os.makedirs(shape_directory)

        for sample_idx in range(num_samples_per_class):
            while True:
                image = np.zeros((image_size, image_size), dtype=np.uint8)
                coords = []

                if shape == 'line':
                    x1, y1 = np.random.randint(0, image_size, size=2)
                    x2, y2 = np.random.randint(0, image_size, size=2)
                    coords = [(x1, y1), (x2, y2)]
                    draw_wobbly_line(image, coords[0], coords[1], color=255, thickness=2, noise_level=3)  # Reduced noise level
                    x_min, y_min, x_max, y_max = get_bounding_box(coords)
                    
                elif shape == 'circle':
                    radius = np.random.randint(20, 60)
                    x, y = np.random.randint(radius, image_size-radius, size=2)
                    coords = [(x, y)]
                    draw_wobbly_circle(image, coords[0], radius, color=255, thickness=2, noise_level=3)  # Reduced noise level
                    x_min, y_min, x_max, y_max = x - radius, y - radius, x + radius, y + radius

                elif shape == 'ellipse':
                    x, y = np.random.randint(30, image_size-30, size=2)
                    axes = np.random.randint(20, 80, size=2)
                    angle = np.random.randint(0, 180)
                    coords = [(x, y)]
                    draw_wobbly_ellipse(image, coords[0], tuple(axes), angle, color=255, thickness=2, noise_level=3)  # Reduced noise level
                    x_min, y_min, x_max, y_max = x - axes[0], y - axes[1], x + axes[0], y + axes[1]

                elif shape == 'rectangle':
                    x1, y1 = np.random.randint(0, image_size-40, size=2)
                    width, height = np.random.randint(30, 80, size=2)
                    x2, y2 = x1 + width, y1 + height
                    coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    if is_within_bounds(coords, image_size, image_size, margin):
                        draw_wobbly_line(image, (x1, y1), (x2, y1), color=255, thickness=2, noise_level=3)  # Reduced noise level
                        draw_wobbly_line(image, (x2, y1), (x2, y2), color=255, thickness=2, noise_level=3)  # Reduced noise level
                        draw_wobbly_line(image, (x2, y2), (x1, y2), color=255, thickness=2, noise_level=3)  # Reduced noise level
                        draw_wobbly_line(image, (x1, y2), (x1, y1), color=255, thickness=2, noise_level=3)  # Reduced noise level
                        x_min, y_min, x_max, y_max = get_bounding_box(coords)
                    else:
                        continue

                elif shape == 'square':
                    x1, y1 = np.random.randint(0, image_size-40, size=2)
                    side_length = np.random.randint(30, 80)
                    x2, y2 = x1 + side_length, y1 + side_length
                    coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    if is_within_bounds(coords, image_size, image_size, margin):
                        draw_wobbly_line(image, (x1, y1), (x2, y1), color=255, thickness=2, noise_level=3)  # Reduced noise level
                        draw_wobbly_line(image, (x2, y1), (x2, y2), color=255, thickness=2, noise_level=3)  # Reduced noise level
                        draw_wobbly_line(image, (x2, y2), (x1, y2), color=255, thickness=2, noise_level=3)  # Reduced noise level
                        draw_wobbly_line(image, (x1, y2), (x1, y1), color=255, thickness=2, noise_level=3)  # Reduced noise level
                        x_min, y_min, x_max, y_max = get_bounding_box(coords)
                    else:
                        continue

                elif shape == 'regular_polygon':
                    num_sides = np.random.randint(5, 8)
                    radius = np.random.randint(30, 60)
                    center = np.random.randint(radius, image_size-radius, size=2)
                    points = []
                    for j in range(num_sides):
                        angle = 2 * np.pi * j / num_sides
                        x = int(center[0] + radius * np.cos(angle))
                        y = int(center[1] + radius * np.sin(angle))
                        points.append((x, y))
                    coords = points
                    if is_within_bounds(coords, image_size, image_size, margin):
                        noisy_points = add_path_noise(coords, noise_level=2)  # Reduced noise level
                        cv2.polylines(image, [np.array(noisy_points)], isClosed=True, color=255, thickness=2)
                        x_min, y_min, x_max, y_max = get_bounding_box(coords)
                    else:
                        continue

                elif shape == 'star':
                    num_points = 5
                    radius_outer = np.random.randint(30, 60)
                    radius_inner = radius_outer // 2
                    center = np.random.randint(radius_outer, image_size-radius_outer, size=2)
                    points = []
                    for j in range(num_points * 2):
                        angle = j * np.pi / num_points
                        radius = radius_outer if j % 2 == 0 else radius_inner
                        x = int(center[0] + radius * np.cos(angle))
                        y = int(center[1] - radius * np.sin(angle))
                        points.append((x, y))
                    coords = points
                    if is_within_bounds(coords, image_size, image_size, margin):
                        noisy_points = add_path_noise(coords, noise_level=2)  # Reduced noise level
                        cv2.polylines(image, [np.array(noisy_points)], isClosed=True, color=255, thickness=2)
                        x_min, y_min, x_max, y_max = get_bounding_box(coords)
                    else:
                        continue

                elif shape == 'triangle':
                    while True:
                        points = np.random.randint(0, image_size, size=(3, 2))
                        coords = points
                        if not (np.linalg.det(np.array([
                            [points[0][0], points[0][1], 1],
                            [points[1][0], points[1][1], 1],
                            [points[2][0], points[2][1], 1]
                        ])) == 0):
                            break
                    if is_within_bounds(coords, image_size, image_size, margin):
                        noisy_points = add_path_noise(coords, noise_level=2)  # Reduced noise level
                        cv2.polylines(image, [np.array(noisy_points)], isClosed=True, color=255, thickness=2)
                        x_min, y_min, x_max, y_max = get_bounding_box(coords)
                    else:
                        continue

                rotation_angle = np.random.uniform(0, 360)
                image = rotate_image(image, rotation_angle)

                # Save image
                image_file_path = os.path.join(shape_directory, f'{shape}_{sample_idx}.png')
                cv2.imwrite(image_file_path, image)

                # Save annotation
                bbox = convert_bbox_to_yolo_format(x_min, y_min, x_max, y_max, image_size, image_size)
                annotation_file_path = os.path.join(shape_directory, f'{shape}_{sample_idx}.txt')
                save_annotation_file(annotation_file_path, shape_id, bbox, image_size, image_size)

                break

generate_geometric_shapes_dataset()
