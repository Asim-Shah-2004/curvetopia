import numpy as np
import matplotlib.pyplot as plt

# Define Point2 and Vector2 as numpy arrays for simplicity
def Point2(x, y):
    return np.array([x, y])

def Vector2(x, y):
    return np.array([x, y])

def V2Add(v1, v2):
    return v1 + v2

def V2Scale(v, s):
    return v * s

def V2Sub(v1, v2):
    return v1 - v2

def V2Distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def V2Dot(v1, v2):
    return np.dot(v1, v2)

def V2Normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

# Bezier curve utility functions
def B0(u): return (1 - u) ** 3
def B1(u): return 3 * u * (1 - u) ** 2
def B2(u): return 3 * u ** 2 * (1 - u)
def B3(u): return u ** 3

# Evaluate a Bezier curve at parameter t
def BezierII(degree, V, t):
    Vtemp = np.copy(V)
    for i in range(1, degree + 1):
        for j in range(degree - i + 1):
            Vtemp[j] = (1.0 - t) * Vtemp[j] + t * Vtemp[j + 1]
    return Vtemp[0]

# Parameterize points by chord length
def ChordLengthParameterize(d):
    u = np.zeros(len(d))
    for i in range(1, len(d)):
        u[i] = u[i - 1] + V2Distance(d[i], d[i - 1])
    u /= u[-1]
    return u

# Generate Bezier curve control points
def GenerateBezier(d, u, tHat1, tHat2):
    C = np.zeros((2, 2))
    X = np.zeros(2)
    A = np.zeros((len(d), 2, 2))

    for i in range(len(d)):
        A[i, 0] = V2Scale(tHat1, B1(u[i]))
        A[i, 1] = V2Scale(tHat2, B2(u[i]))
        C[0][0] += V2Dot(A[i, 0], A[i, 0])
        C[0][1] += V2Dot(A[i, 0], A[i, 1])
        C[1][0] = C[0][1]
        C[1][1] += V2Dot(A[i, 1], A[i, 1])

        tmp = V2Sub(d[i], V2Add(V2Add(V2Add(
            V2Scale(d[0], B0(u[i])),
            V2Scale(d[0], B1(u[i]))),
            V2Scale(d[-1], B2(u[i]))),
            V2Scale(d[-1], B3(u[i]))))
        X[0] += V2Dot(A[i, 0], tmp)
        X[1] += V2Dot(A[i, 1], tmp)

    det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1]
    det_C0_X = C[0][0] * X[1] - C[0][1] * X[0]
    det_X_C1 = X[0] * C[1][1] - X[1] * C[0][1]

    alpha_l = det_X_C1 / det_C0_C1
    alpha_r = det_C0_X / det_C0_C1

    segLength = V2Distance(d[0], d[-1])
    epsilon = 1.0e-6 * segLength
    if alpha_l < epsilon or alpha_r < epsilon:
        alpha_l = alpha_r = segLength / 3.0

    bezCurve = np.zeros((4, 2))
    bezCurve[0] = d[0]
    bezCurve[3] = d[-1]
    bezCurve[1] = V2Add(bezCurve[0], V2Scale(tHat1, alpha_l))
    bezCurve[2] = V2Add(bezCurve[3], V2Scale(tHat2, alpha_r))
    return bezCurve

# Function to fit a Bezier curve to hand-drawn points
def fit_bezier_curve(hand_drawn_points):
    u = ChordLengthParameterize(hand_drawn_points)
    tHat1 = V2Normalize(V2Sub(hand_drawn_points[1], hand_drawn_points[0]))
    tHat2 = V2Normalize(V2Sub(hand_drawn_points[-2], hand_drawn_points[-1]))
    bezCurve = GenerateBezier(hand_drawn_points, u, tHat1, tHat2)
    return bezCurve

# Plot the Bezier curve and the original data points
def plot_curve(d, bezCurve):
    # Plot original points
    plt.plot(d[:, 0], d[:, 1], 'ro-', label='Original Points')
    
    # Plot Bezier curve
    t = np.linspace(0, 1, 100)
    bezier_points = np.array([BezierII(3, bezCurve, t_val) for t_val in t])
    plt.plot(bezier_points[:, 0], bezier_points[:, 1], 'b-', label='Fitted Bezier Curve')
    
    plt.legend()
    plt.show()

# Read CSV and extract paths
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

# Main function to read CSV, fit curves, and plot the result
def main(csv_path):
    paths = read_csv(csv_path)
    for path in paths:
        for hand_drawn_points in path:
            bezCurve = fit_bezier_curve(hand_drawn_points)
            plot_curve(hand_drawn_points, bezCurve)

# Example usage
csv_path = '../problems/frag0.csv'
main(csv_path)
