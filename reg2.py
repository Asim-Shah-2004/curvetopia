import numpy as np
import matplotlib.pyplot as plt
import svgwrite
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

def fit_line(points):
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    return slope, intercept

def fit_circle(points):
    def calc_R(xc, yc):
        return np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2)

    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = np.mean(points, axis=0)
    center_2, ier = minimize(lambda c: np.sum(f_2(c)**2), center_estimate).x
    radius = calc_R(*center_2).mean()
    return center_2, radius

def create_svg(points, lines=[], circles=[]):
    dwg = svgwrite.Drawing('output.svg', profile='tiny')
    for line in lines:
        slope, intercept = line
        start = (points[0, 0], slope * points[0, 0] + intercept)
        end = (points[-1, 0], slope * points[-1, 0] + intercept)
        dwg.add(dwg.line(start=start, end=end, stroke=svgwrite.rgb(10, 10, 16, '%')))

    for circle in circles:
        center, radius = circle
        dwg.add(dwg.circle(center=center, r=radius, stroke=svgwrite.rgb(10, 10, 16, '%'), fill='none'))

    dwg.save()

def regularize_shapes(points):
    lines = []
    circles = []
    # Example condition to check if points form a line
    if len(points) > 2:
        slope, intercept = fit_line(points)
        lines.append((slope, intercept))
    if len(points) > 3:
        center, radius = fit_circle(points)
        circles.append((center, radius))
    return lines, circles

def main():
    # Example input
    points = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5]
    ])

    lines, circles = regularize_shapes(points)
    create_svg(points, lines=lines, circles=circles)

if __name__ == "__main__":
    main()
