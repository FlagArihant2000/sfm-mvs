import cv2
import numpy as np
import os
from scipy.optimize import least_squares



def OptimReprojectionError(X_locs, p, r, t, K):
    total_error = 0
    p = p.T
    num_pts = len(p)
    R = X_locs[0:9].reshape((3, 3))
    t = X_locs[9:12]
    K = X_locs[12:21].reshape((3, 3))
    X_locs = np.float32(X_locs[21:].reshape((num_pts, 1, 3)))
    error = []
    r, _ = cv2.Rodrigues(R)
    p2d, _ = cv2.projectPoints(X_locs, r, t, K, distCoeffs=None)
    p2d = p2d[:, 0, :]
    # p, _ = cv2.projectPoints(X, r, t, K, distCoeffs = None)
    for idx in range(num_pts):
        img_pt = p[idx]
        reprojected_pt = p2d[idx]
        er = (img_pt - reprojected_pt) ** 2
        error.append(er)

    return np.array(error).ravel() / num_pts


def BundleAdjustment(X, pts1, P, Rt, K):
    print(X.shape, pts1.shape)
    num_points = len(pts1.T)
    R = Rt[:3, :3]
    t = Rt[:3, 3]
    opt_variables = np.hstack((R.ravel(), t.ravel()))
    opt_variables = np.hstack((opt_variables, K.ravel()))
    opt_variables = np.hstack((opt_variables, X.ravel()))
    print("The Size of opt_variables=", opt_variables.shape)

    corrected_values = least_squares(OptimReprojectionError, opt_variables, args=(pts1, num_points, t, K))

    corrected_values = corrected_values.x
    R = corrected_values[0:9].reshape((3, 3))
    t = corrected_values[9:12].reshape((3, 1))
    K = corrected_values[12:21].reshape((3, 3))
    points_3d = corrected_values[21:].reshape((num_points, 1, 3))
    Rt = np.hstack((R, t))
    P = np.matmul(K, Rt)

    return Rt, P, points_3d
