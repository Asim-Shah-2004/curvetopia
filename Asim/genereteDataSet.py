import numpy as np
import cv2
import os
import random

# Define image size and shape properties
IMG_SIZE = 128
SHAPE_PROPERTIES = {
    'line': {'num_points': 2},
    'triangle': {'num_points': 3},
    'square': {'num_points': 4},
    'circle': {'radius_range': (10, 40)},
    'ellipse': {'axes_range': ((10, 30), (10, 40))},
    'rectangle': {'num_points': 4},
    'star': {'num_points': 5, 'inner_radius': 10, 'outer_radius': 20},
    'regular_polygon': {'num_points': 6}
}

# Helper functions to create shapes
def add_noise_to_points(points, noise_level=10):
    noisy_points = []
    for (x, y) in points:
        noisy_points.append((x + random.randint(-noise_level, noise_level),
                             y + random.randint(-noise_level, noise_level)))
    return noisy_points

def draw_hand_drawn_shape(image, shape, properties):
    if shape == 'line':
        points = [(random.randint(10, IMG_SIZE-10), random.randint(10, IMG_SIZE-10)) for _ in range(properties['num_points'])]
        points = add_noise_to_points(points)
        cv2.line(image, points[0], points[1], (255), thickness=random.randint(1, 3))
    elif shape == 'triangle' or shape == 'square' or shape == 'rectangle':
        points = [(random.randint(10, IMG_SIZE-10), random.randint(10, IMG_SIZE-10)) for _ in range(properties['num_points'])]
        points = add_noise_to_points(points)
        cv2.polylines(image, [np.array(points)], isClosed=True, color=(255), thickness=random.randint(1, 3))
    elif shape == 'circle':
        center = (random.randint(20, IMG_SIZE-20), random.randint(20, IMG_SIZE-20))
        radius = random.randint(properties['radius_range'][0], properties['radius_range'][1])
        radius = radius + random.randint(-5, 5)
        cv2.circle(image, center, radius, (255), thickness=random.randint(1, 3))
    elif shape == 'ellipse':
        center = (random.randint(20, IMG_SIZE-20), random.randint(20, IMG_SIZE-20))
        axes = (random.randint(properties['axes_range'][0][0], properties['axes_range'][0][1]),
                random.randint(properties['axes_range'][1][0], properties['axes_range'][1][1]))
        angle = random.randint(0, 360)
        axes = (axes[0] + random.randint(-5, 5), axes[1] + random.randint(-5, 5))
        cv2.ellipse(image, center, axes, angle, 0, 360, (255), thickness=random.randint(1, 3))
    elif shape == 'star':
        center = (random.randint(20, IMG_SIZE-20), random.randint(20, IMG_SIZE-20))
        inner_radius = properties['inner_radius'] + random.randint(-5, 5)
        outer_radius = properties['outer_radius'] + random.randint(-5, 5)
        points = []
        for i in range(10):
            angle = i * np.pi / 5
            r = inner_radius if i % 2 == 0 else outer_radius
            x = center[0] + int(np.cos(angle) * r)
            y = center[1] + int(np.sin(angle) * r)
            points.append((x, y))
        points = add_noise_to_points(points)
        cv2.polylines(image, [np.array(points)], isClosed=True, color=(255), thickness=random.randint(1, 3))
    elif shape == 'regular_polygon':
        center = (random.randint(20, IMG_SIZE-20), random.randint(20, IMG_SIZE-20))
        radius = 20 + random.randint(-5, 5)
        points = []
        for i in range(properties['num_points']):
            angle = 2 * np.pi * i / properties['num_points']
            x = center[0] + int(np.cos(angle) * radius)
            y = center[1] + int(np.sin(angle) * radius)
            points.append((x, y))
        points = add_noise_to_points(points)
        cv2.polylines(image, [np.array(points)], isClosed=True, color=(255), thickness=random.randint(1, 3))
    return image

def draw_smooth_shape(image, shape, properties):
    if shape == 'line':
        points = [(random.randint(10, IMG_SIZE-10), random.randint(10, IMG_SIZE-10)) for _ in range(properties['num_points'])]
        cv2.line(image, points[0], points[1], (255), thickness=2)
    elif shape == 'triangle' or shape == 'square' or shape == 'rectangle':
        points = [(random.randint(10, IMG_SIZE-10), random.randint(10, IMG_SIZE-10)) for _ in range(properties['num_points'])]
        cv2.polylines(image, [np.array(points)], isClosed=True, color=(255), thickness=2)
    elif shape == 'circle':
        center = (random.randint(20, IMG_SIZE-20), random.randint(20, IMG_SIZE-20))
        radius = random.randint(properties['radius_range'][0], properties['radius_range'][1])
        cv2.circle(image, center, radius, (255), thickness=2)
    elif shape == 'ellipse':
        center = (random.randint(20, IMG_SIZE-20), random.randint(20, IMG_SIZE-20))
        axes = (random.randint(properties['axes_range'][0][0], properties['axes_range'][0][1]),
                random.randint(properties['axes_range'][1][0], properties['axes_range'][1][1]))
        angle = random.randint(0, 360)
        cv2.ellipse(image, center, axes, angle, 0, 360, (255), thickness=2)
    elif shape == 'star':
        center = (random.randint(20, IMG_SIZE-20), random.randint(20, IMG_SIZE-20))
        inner_radius = properties['inner_radius']
        outer_radius = properties['outer_radius']
        points = []
        for i in range(10):
            angle = i * np.pi / 5
            r = inner_radius if i % 2 == 0 else outer_radius
            x = center[0] + int(np.cos(angle) * r)
            y = center[1] + int(np.sin(angle) * r)
            points.append((x, y))
        cv2.polylines(image, [np.array(points)], isClosed=True, color=(255), thickness=2)
    elif shape == 'regular_polygon':
        center = (random.randint(20, IMG_SIZE-20), random.randint(20, IMG_SIZE-20))
        radius = 20
        points = []
        for i in range(properties['num_points']):
            angle = 2 * np.pi * i / properties['num_points']
            x = center[0] + int(np.cos(angle) * radius)
            y = center[1] + int(np.sin(angle) * radius)
            points.append((x, y))
        cv2.polylines(image, [np.array(points)], isClosed=True, color=(255), thickness=2)
    return image

# Generate dataset
def generate_dataset(output_dir, num_samples=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    hand_drawn_dir = os.path.join(output_dir, 'hand_drawn')
    smooth_dir = os.path.join(output_dir, 'smooth')
    if not os.path.exists(hand_drawn_dir):
        os.makedirs(hand_drawn_dir)
    if not os.path.exists(smooth_dir):
        os.makedirs(smooth_dir)

    for i in range(num_samples):
        shape = random.choice(list(SHAPE_PROPERTIES.keys()))
        properties = SHAPE_PROPERTIES[shape]

        hand_drawn_image = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        smooth_image = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        hand_drawn_image = draw_hand_drawn_shape(hand_drawn_image, shape, properties)
        smooth_image = draw_smooth_shape(smooth_image, shape, properties)

        cv2.imwrite(os.path.join(hand_drawn_dir, f'{i}_{shape}.png'), hand_drawn_image)
        cv2.imwrite(os.path.join(smooth_dir, f'{i}_{shape}.png'), smooth_image)

# Generate the dataset
generate_dataset('shapes_dataset')
