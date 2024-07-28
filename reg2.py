import numpy as np
import matplotlib.pyplot as plt

class FitCurves:
    @staticmethod
    def fit_curve(points, error):
        def chord_length_parameterize(points):
            u = [0]
            for i in range(1, len(points)):
                u.append(u[-1] + np.linalg.norm(points[i] - points[i - 1]))
            return [x / u[-1] for x in u]

        def generate_bezier(points, u):
            A = np.zeros((len(u), 2, 2))
            C = np.zeros((2, 2))
            X = np.zeros(2)

            for i in range(len(u)):
                A[i][0] = (1 - u[i]) ** 2 * points[0] + (1 - u[i]) * u[i] * 2 * points[1]
                A[i][1] = (1 - u[i]) * u[i] * 2 * points[2] + u[i] ** 2 * points[3]

            for i in range(len(u)):
                C[0][0] += np.dot(A[i][0], A[i][0])
                C[0][1] += np.dot(A[i][0], A[i][1])
                C[1][0] += np.dot(A[i][1], A[i][0])
                C[1][1] += np.dot(A[i][1], A[i][1])

                tmp = points[i] - (1 - u[i]) ** 3 * points[0] - (1 - u[i]) ** 2 * u[i] * 3 * points[1] - (1 - u[i]) * u[i] ** 2 * 3 * points[2] - u[i] ** 3 * points[3]
                X[0] += np.dot(A[i][0], tmp)
                X[1] += np.dot(A[i][1], tmp)

            det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1]
            if abs(det_C0_C1) < 1e-10:
                return [points[0], points[0], points[-1], points[-1]]

            alpha_l = (C[1][1] * X[0] - C[0][1] * X[1]) / det_C0_C1
            alpha_r = (C[0][0] * X[1] - C[1][0] * X[0]) / det_C0_C1

            seg_length = np.linalg.norm(points[0] - points[-1])
            epsilon = 1e-6 * seg_length

            if alpha_l < epsilon or alpha_r < epsilon:
                alpha_l = alpha_r = seg_length / 3.0

            return [points[0], points[0] + alpha_l * (points[1] - points[0]), points[-1] + alpha_r * (points[-2] - points[-1]), points[-1]]

        def bezier_point(t, bezier):
            return (1 - t) ** 3 * bezier[0] + 3 * (1 - t) ** 2 * t * bezier[1] + 3 * (1 - t) * t ** 2 * bezier[2] + t ** 3 * bezier[3]

        def fit_cubic(points, u, t1, t2, error, depth=0, max_depth=10):
            if depth > max_depth:
                return []

            bezier = generate_bezier(points, u)
            max_dist, split_point = max((np.linalg.norm(p - bezier_point(t, bezier)), i) for i, (p, t) in enumerate(zip(points, u)))
            if max_dist < error:
                return [bezier]
            left_u = u[:split_point + 1]
            right_u = [v - u[split_point] for v in u[split_point:]]
            return fit_cubic(points[:split_point + 1], left_u, t1, np.array([0, 0]), error, depth+1) + fit_cubic(points[split_point:], right_u, np.array([0, 0]), t2, depth+1)

        points = np.array(points)
        u = chord_length_parameterize(points)
        return fit_cubic(points, u, np.array([0, 0]), np.array([0, 0]), error)

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

def plot(paths_XYs, title):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.title(title)
    plt.show()

def fit_bezier_curve(points, error=0.01):
    curves = FitCurves.fit_curve(points, error)
    bezier_points = []
    for curve in curves:
        t = np.linspace(0, 1, 100)
        bezier = np.array([(1 - t)**3 * curve[0] + 3 * (1 - t)**2 * t * curve[1] + 3 * (1 - t) * t**2 * curve[2] + t**3 * curve[3] for t in t])
        bezier_points.append(bezier)
    return np.vstack(bezier_points)

def process_points(input_points, error=0.01):
    output_points = []
    for paths in input_points:
        fitted_paths = []
        for points in paths:
            if len(points) > 1:
                fitted_points = fit_bezier_curve(points, error)
                fitted_paths.append(fitted_points)
            else:
                fitted_paths.append(points)
        output_points.append(fitted_paths)
    return output_points

# Define colours
colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Read input file
input_path = "problems/frag0.csv"
input_points = read_csv(input_path)

# Plot input points
plot(input_points, "Input Points")

# Process points to fit Bézier curves
error = 0.01  # Define your desired error tolerance
output_points = process_points(input_points, error)

# Plot output points
plot(output_points, "Output Points")

# Save output points to CSV
output_path = "/mnt/data/frag01_sol.csv"
with open(output_path, 'w') as f:
    for path_idx, paths in enumerate(output_points):
        for line_idx, line in enumerate(paths):
            for point in line:
                f.write(f"{path_idx},{line_idx},{point[0]},{point[1]}\n")

print(f"Output saved to {output_path}")
