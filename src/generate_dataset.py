import numpy as np
import cv2
import os


def rotate_image(image, rotation_angle):
    """
    Rotate an image by a specified angle around its center.

    Parameters:
    - image: The input image to be rotated (numpy array).
    - rotation_angle: The angle by which to rotate the image (in degrees).

    Returns:
    - Rotated image (numpy array).
    """
    image_center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    return rotated_image


def is_within_bounds(coords, width, height, margin=0):
    """
    Check if all coordinates are within the image boundaries with an optional margin.

    Parameters:
    - coords: List of coordinates to be checked.
    - width: Image width.
    - height: Image height.
    - margin: Margin to be considered inside the boundary.

    Returns:
    - Boolean indicating whether all coordinates are within bounds.
    """
    return all(0 + margin <= x < width - margin and 0 + margin <= y < height - margin for x, y in coords)


def add_path_noise(points, noise_level=2):
    """
    Add random noise to the points of a shape to simulate a hand-drawn effect.

    Parameters:
    - points: List of points to which noise will be added.
    - noise_level: The level of noise to be added to each point.

    Returns:
    - Noisy points (list of tuples).
    """
    noisy_points = []
    for (x, y) in points:
        noisy_x = x + np.random.uniform(-noise_level, noise_level)
        noisy_y = y + np.random.uniform(-noise_level, noise_level)
        noisy_points.append((int(noisy_x), int(noisy_y)))
    return noisy_points


def draw_rounded_rectangle(img, top_left, bottom_right, radius, color, thickness, noise_level):
    """
    Draw a rounded rectangle with smooth corners on an image, with noise added to the path.

    Parameters:
    - img: The image on which to draw the shape.
    - top_left: Coordinates of the top-left corner of the rectangle.
    - bottom_right: Coordinates of the bottom-right corner of the rectangle.
    - radius: Radius of the rounded corners.
    - color: Color of the shape.
    - thickness: Thickness of the shape's boundary.
    - noise_level: Level of noise to be added to the shape path.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Draw rectangle sides with noise
    rect_coords = [
        (x1 + radius, y1), (x2 - radius, y1),
        (x2 - radius, y2), (x1 + radius, y2),
        (x1 + radius, y1 + radius)
    ]
    noisy_rect_coords = add_path_noise(rect_coords, noise_level)
    for i in range(4):
        cv2.line(img, noisy_rect_coords[i], noisy_rect_coords[i + 1], color, thickness)

    # Draw the rounded corners with noise
    corners = [
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x2 - radius, y2 - radius),
        (x1 + radius, y2 - radius)
    ]
    noisy_corners = add_path_noise(corners, noise_level)
    for i, (center, angle_start, angle_end) in enumerate(zip(noisy_corners, [180, 270, 0, 90], [90, 0, 90, 180])):
        cv2.ellipse(img, center, (radius, radius), angle_start, 0, angle_end, color, thickness)


def generate_geometric_shapes_dataset(num_samples_per_class=1500, output_directory='../dataset'):
    """
    Generate a dataset of synthetic images containing various geometric shapes.

    Parameters:
    - num_samples_per_class: Number of images to generate for each shape category.
    - output_directory: Directory where the generated images will be saved.
    """
    shape_categories = ['line', 'triangle', 'square', 'circle', 'rectangle', 'star', 'regular_polygon']
    image_size = 224
    margin = 10

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for shape in shape_categories:
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
                    noisy_coords = add_path_noise(coords, noise_level=4)
                    cv2.line(image, noisy_coords[0], noisy_coords[1], color=255, thickness=2)

                elif shape == 'circle':
                    radius = np.random.randint(20, 60)
                    x, y = np.random.randint(radius, image_size-radius, size=2)
                    coords = [(x, y)]
                    cv2.circle(image, coords[0], radius, color=255, thickness=2)

                elif shape == 'ellipse':
                    x, y = np.random.randint(30, image_size-30, size=2)
                    axes = np.random.randint(20, 80, size=2)
                    angle = np.random.randint(0, 180)
                    coords = [(x, y)]
                    cv2.ellipse(image, coords[0], tuple(axes), angle, 0, 360, color=255, thickness=2)

                elif shape == 'rectangle':
                    x1, y1 = np.random.randint(0, image_size-40, size=2)
                    width, height = np.random.randint(30, 80, size=2)
                    x2, y2 = x1 + width, y1 + height
                    coords = [(x1, y1), (x2, y2)]
                    if is_within_bounds(coords, image_size, image_size, margin):
                        noisy_coords = add_path_noise(coords, noise_level=4)
                        cv2.rectangle(image, noisy_coords[0], noisy_coords[1], color=255, thickness=2)
                    else:
                        continue

                elif shape == 'square':
                    x1, y1 = np.random.randint(0, image_size-40, size=2)
                    side_length = np.random.randint(30, 80)
                    x2, y2 = x1 + side_length, y1 + side_length
                    coords = [(x1, y1), (x2, y2)]
                    if is_within_bounds(coords, image_size, image_size, margin):
                        noisy_coords = add_path_noise(coords, noise_level=4)
                        cv2.rectangle(image, noisy_coords[0], noisy_coords[1], color=255, thickness=2)
                    else:
                        continue

                elif shape == 'rounded_rectangle':
                    x1, y1 = np.random.randint(0, image_size-40, size=2)
                    width, height = np.random.randint(30, 80, size=2)
                    radius = np.random.randint(10, 30)
                    x2, y2 = x1 + width, y1 + height
                    coords = [(x1, y1), (x2, y2)]
                    if is_within_bounds(coords, image_size, image_size, margin):
                        draw_rounded_rectangle(image, (x1, y1), (x2, y2), radius, color=255, thickness=2, noise_level=4)
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
                        noisy_points = add_path_noise(coords, noise_level=4)
                        cv2.polylines(image, [np.array(noisy_points)], isClosed=True, color=255, thickness=2)
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
                        noisy_points = add_path_noise(coords, noise_level=4)
                        cv2.polylines(image, [np.array(noisy_points)], isClosed=True, color=255, thickness=2)
                    else:
                        continue

                elif shape == 'triangle':
                    while True:
                        points = np.random.randint(0, image_size, size=(3, 2))
                        coords = points
                        # Check if the points form a valid triangle (i.e., not collinear)
                        if not (np.linalg.det(np.array([
                            [points[0][0], points[0][1], 1],
                            [points[1][0], points[1][1], 1],
                            [points[2][0], points[2][1], 1]
                        ])) == 0):
                            break
                    if is_within_bounds(coords, image_size, image_size, margin):
                        noisy_points = add_path_noise(coords, noise_level=4)
                        cv2.polylines(image, [np.array(noisy_points)], isClosed=True, color=255, thickness=2)
                    else:
                        continue

                # Apply random rotation
                rotation_angle = np.random.uniform(0, 360)
                image = rotate_image(image, rotation_angle)

                # Save the generated image
                file_path = os.path.join(shape_directory, f'{shape}_{sample_idx}.png')
                cv2.imwrite(file_path, image)
                break  # Break out of the while loop after successfully generating an image


# Generate the dataset
generate_geometric_shapes_dataset()
