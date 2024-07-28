import numpy as np
from numpy.linalg import norm, solve
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class FitCurves:
    @staticmethod
    def fit_curve(points, error):
        """
        Fit a Bezier curve to a set of points with a specified error tolerance.
        """
        tHat1 = FitCurves._compute_left_tangent(points, 0)
        tHat2 = FitCurves._compute_right_tangent(points, len(points) - 1)
        result = []
        FitCurves._fit_cubic(points, 0, len(points) - 1, tHat1, tHat2, error, result)
        return result

    @staticmethod
    def _fit_cubic(points, first, last, tHat1, tHat2, error, result):
        """
        Fit a cubic Bezier curve to a subset of points.
        """
        nPts = last - first + 1
        if nPts == 2:
            dist = norm(points[first] - points[last]) / 3.0
            bezCurve = [
                points[first],
                points[first] + tHat1 * dist,
                points[last] + tHat2 * dist,
                points[last]
            ]
            result.extend(bezCurve[1:])
            return

        u = FitCurves._chord_length_parameterize(points, first, last)
        bezCurve = FitCurves._generate_bezier(points, first, last, u, tHat1, tHat2)

        maxError, splitPoint = FitCurves._compute_max_error(points, first, last, bezCurve, u)
        if maxError < error:
            result.extend(bezCurve[1:])
            return

        if maxError < error ** 2:
            for _ in range(4):
                uPrime = FitCurves._reparameterize(points, first, last, u, bezCurve)
                bezCurve = FitCurves._generate_bezier(points, first, last, uPrime, tHat1, tHat2)
                maxError, splitPoint = FitCurves._compute_max_error(points, first, last, bezCurve, uPrime)
                if maxError < error:
                    result.extend(bezCurve[1:])
                    return
                u = uPrime

        tHatCenter = FitCurves._compute_center_tangent(points, splitPoint)
        FitCurves._fit_cubic(points, first, splitPoint, tHat1, tHatCenter, error, result)
        tHatCenter = -tHatCenter
        FitCurves._fit_cubic(points, splitPoint, last, tHatCenter, tHat2, error, result)

    @staticmethod
    def _generate_bezier(points, first, last, uPrime, tHat1, tHat2):
        """
        Generate a Bezier curve approximation to a set of points.
        """
        nPts = last - first + 1
        A = np.zeros((nPts, 2, 2))
        C = np.zeros((2, 2))
        X = np.zeros((2, 1))

        for i in range(nPts):
            A[i][0] = tHat1 * FitCurves._B1(uPrime[i])
            A[i][1] = tHat2 * FitCurves._B2(uPrime[i])

        for i in range(nPts):
            C[0][0] += np.dot(A[i][0], A[i][0])
            C[0][1] += np.dot(A[i][0], A[i][1])
            C[1][0] = C[0][1]
            C[1][1] += np.dot(A[i][1], A[i][1])

            tmp = points[first + i] - (
                points[first] * FitCurves._B0(uPrime[i]) +
                points[first] * FitCurves._B1(uPrime[i]) +
                points[last] * FitCurves._B2(uPrime[i]) +
                points[last] * FitCurves._B3(uPrime[i])
            )

            X[0][0] += np.dot(A[i][0], tmp)
            X[1][0] += np.dot(A[i][1], tmp)

        det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1]
        if det_C0_C1 == 0:
            alpha_l, alpha_r = 0, 0
        else:
            det_C0_X = C[0][0] * X[1][0] - C[1][0] * X[0][0]
            det_X_C1 = X[0][0] * C[1][1] - X[1][0] * C[0][1]
            alpha_l = det_X_C1 / det_C0_C1
            alpha_r = det_C0_X / det_C0_C1

        segLength = norm(points[first] - points[last])
        epsilon = 1.0e-6 * segLength
        if alpha_l < epsilon or alpha_r < epsilon:
            dist = segLength / 3.0
            bezCurve = [
                points[first],
                points[first] + tHat1 * dist,
                points[last] + tHat2 * dist,
                points[last]
            ]
            return bezCurve

        bezCurve = [
            points[first],
            points[first] + tHat1 * alpha_l,
            points[last] + tHat2 * alpha_r,
            points[last]
        ]
        return bezCurve

    @staticmethod
    def _reparameterize(points, first, last, u, bezCurve):
        """
        Reparameterize points and fit the Bezier curve.
        """
        uPrime = [FitCurves._newton_raphson_root_find(bezCurve, points[i], u[i - first])
                  for i in range(first, last + 1)]
        return uPrime

    @staticmethod
    def _newton_raphson_root_find(bez, point, u):
        """
        Use Newton-Raphson iteration to find a better root.
        """
        Q_u = FitCurves._bezier_ii(3, bez, u)
        Q1 = [3 * (bez[i + 1] - bez[i]) for i in range(3)]
        Q2 = [2 * (Q1[i + 1] - Q1[i]) for i in range(2)]
        Q1_u = FitCurves._bezier_ii(2, Q1, u)
        Q2_u = FitCurves._bezier_ii(1, Q2, u)
        numerator = np.dot(Q_u - point, Q1_u)
        denominator = np.dot(Q1_u, Q1_u) + np.dot(Q_u - point, Q2_u)
        if denominator == 0:
            return u
        return u - numerator / denominator

    @staticmethod
    def _bezier_ii(degree, V, t):
        """
        Evaluate a Bezier curve at a particular parameter value.
        """
        Vtemp = np.copy(V)
        for i in range(1, degree + 1):
            for j in range(degree - i + 1):
                Vtemp[j] = (1 - t) * Vtemp[j] + t * Vtemp[j + 1]
        return Vtemp[0]

    @staticmethod
    def _B0(u):
        return (1 - u) ** 3

    @staticmethod
    def _B1(u):
        return 3 * u * (1 - u) ** 2

    @staticmethod
    def _B2(u):
        return 3 * u ** 2 * (1 - u)

    @staticmethod
    def _B3(u):
        return u ** 3

    @staticmethod
    def _compute_left_tangent(points, end):
        tHat1 = points[end + 1] - points[end]
        return tHat1 / norm(tHat1)

    @staticmethod
    def _compute_right_tangent(points, end):
        tHat2 = points[end - 1] - points[end]
        return tHat2 / norm(tHat2)

    @staticmethod
    def _compute_center_tangent(points, center):
        V1 = points[center - 1] - points[center]
        V2 = points[center] - points[center + 1]
        tHatCenter = (V1 + V2) / 2.0
        return tHatCenter / norm(tHatCenter)

    @staticmethod
    def _chord_length_parameterize(points, first, last):
        u = [0.0]
        for i in range(first + 1, last + 1):
            u.append(u[-1] + norm(points[i] - points[i - 1]))
        for i in range(1, len(u)):
            u[i] /= u[-1]
        return u

    @staticmethod
    def _compute_max_error(points, first, last, bezCurve, u):
        maxDist = 0.0
        splitPoint = (last - first + 1) // 2
        for i in range(first + 1, last):
            P = FitCurves._bezier_ii(3, bezCurve, u[i - first])
            v = P - points[i]
            dist = np.dot(v, v)
            if dist > maxDist:
                maxDist = dist
                splitPoint = i
        return maxDist, splitPoint


# Example set of points
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

# Fit curve with specified error tolerance

points = read_csv('problems/isolated.csv')
points = np.array(points[2][0]) #do 00 10 20 to change shapes


error = 0.01
fit = FitCurves.fit_curve(points, error)

# Plot the original points
plt.plot(points[:, 0], points[:, 1], 'ro-', label='Original Points')

# Plot the fitted Bezier curve
fit = np.array(fit).reshape(-1, 2)
plt.plot(fit[:, 0], fit[:, 1], 'b-', label='Fitted Bezier Curve')

plt.legend()
plt.show()
