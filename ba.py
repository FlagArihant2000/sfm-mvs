import numpy as np
import cv2
from scipy.optimize import least_squares

# We need camera parameters (n_camera, 9)
# Points 3D (n_points, 3)
# Camera index (n_observations, )
# Point index (n_observations, )
# 2D representation of points (n_observations, 2)

# For camera parameters, first three are the rotation vector
# Other 3 are translation parameters
# Other three are focal length and two distortion parameters

# Step 1: Rotate points, with given rotation parameters

def ReprojectionError(X, pts, K, R, t):
	total_error = 0
	
	r, _ = cv2.Rodrigues(R)
	
	#print(X.shape)
	
	p, _ = cv2.projectPoints(X, r, t, K, distCoeffs = None)
	
	p = p[:, 0, :]
	
	#error = cv2.norm(p, pts, cv2.NORM_L2)/len(p)
	
	r_error = []
	for x in range(len(p)):
		error = (p[x][0] - pts[x][0])**2 + (p[x][1] - pts[x][1])**2
		r_error = r_error + [error]
	
	r_error = np.array(r_error)
	r_error = r_error/len(p)
	
	tot_error = r_error ** 2
	tot_error = tot_error / len(X)
	
	return tot_error
	

def bundle_adjustment(X, p, img, K, R, t):
	num_points = len(p)
	tot_error = ReprojectionError(X, p, K, R, t)
	print(len(tot_error), len(X.ravel()))
	P = np.matmul(K, np.hstack((R,t)))
	opt_variables = np.hstack((P.ravel(), X.ravel(order = "F")))
	corrected_values = least_squares(ReprojectionError, opt_variables, args = (p, num_points))
	#print(corrected_values)
	
	
	
	
	
	
