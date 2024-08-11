import numpy as np
import cv2
import matplotlib.pyplot as plt
import svgwrite
import cairosvg
import os
from typing import List, Tuple

temp_dir = '../temp'
os.makedirs(f'{temp_dir}', exist_ok=True)


def read_csv(csv_path: str) -> List[List[np.ndarray]]:
    """
    Reads a CSV file and processes it into a list of paths with coordinates.
    Each path consists of a list of coordinates representing a shape.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        List[List[np.ndarray]]: A nested list where each sub-list contains numpy arrays of coordinates for each shape.
    """
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


def polylines2svg(XY: np.ndarray, svg_path: str, colour: str = 'blue', stroke_width: int = 2) -> Tuple[str, str]:
    """
    Converts a single path to an SVG file and then to a PNG file with outline only.

    Args:
        XY (np.ndarray): A 2D numpy array of shape (rows, 2) where rows are points with X and Y coordinates.
        svg_path (str): The path where the SVG file will be saved.
        colour (str): The color to use for the path outline.
        stroke_width (int): The width of the path outline.

    Returns:
        Tuple[str, str]: The paths to the SVG and PNG files.
    """
    if XY.ndim != 2 or XY.shape[1] != 2:
        raise ValueError("XY should be a 2D array with shape (rows, 2)")

    # Determine the width and height of the SVG
    W = np.max(XY[:, 0]) + 10
    H = np.max(XY[:, 1]) + 10

    # Create a new SVG drawing
    dwg = svgwrite.Drawing(svg_path, profile='tiny', size=(W, H))
    dwg.viewbox(0, 0, W, H)
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill='white'))

    # Create the path data for SVG
    path_data = [("M", (XY[0, 0], XY[0, 1]))]
    for i in range(1, len(XY)):
        path_data.append(("L", (XY[i, 0], XY[i, 1])))
    # path_data.append(("Z", None))  # Uncomment if you need to close the path

    dwg.add(dwg.path(d=path_data, fill='none', stroke=colour, stroke_width=stroke_width))
    dwg.save()

    # Convert SVG to PNG
    png_path = svg_path.replace('.svg', '.png')
    cairosvg.svg2png(url=svg_path, write_to=png_path, parent_width=W, parent_height=H,
                     output_width=W, output_height=H, background_color='white')

    return svg_path, png_path


def classify_and_plot(path_XYs: List[List[np.ndarray]]) -> None:
    """
    Classifies and plots shapes from given path data.

    Args:
        path_XYs (List[List[np.ndarray]]): A list of lists where each sub-list contains numpy arrays of coordinates for each shape.

    Returns:
        None
    """
    colours = ['blue', 'red', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'gray']

    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(path_XYs):
        c = colours[i % len(colours)]
        for j, XY in enumerate(XYs):
            XY = np.array(XY)
            svg_path, png_path = polylines2svg(XY, f"{temp_dir}/curve_{i}_{j}.svg", colour='blue')
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)

    ax.set_aspect('equal')
    plt.show()


# Example usage
path_XYs = read_csv('../problems/regularized_shapes.csv')
classify_and_plot(path_XYs)
