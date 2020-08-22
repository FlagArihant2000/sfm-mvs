import cv2
import numpy as np

# Set up two cameras near each other

K = np.array([
    [718.856 ,   0.  ,   607.1928],
    [  0.  ,   718.856 , 185.2157],
    [  0.  ,     0.   ,    1.    ],
])

R1 = np.array([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.]
])

R2 = np.array([
    [ 0.99999183 ,-0.00280829 ,-0.00290702],
    [ 0.0028008  , 0.99999276, -0.00257697],
    [ 0.00291424 , 0.00256881 , 0.99999245]
])

t1 = np.array([[0.], [0.], [0.]])

t2 = np.array([[-0.02182627], [ 0.00733316], [ 0.99973488]])

P1 = np.hstack([R1.T, -R1.T.dot(t1)])
P2 = np.hstack([R2.T, -R2.T.dot(t2)])

P1 = K.dot(P1)
P2 = K.dot(P2)

# Corresponding image points
imagePoint1 = np.array([371.91915894, 221.53485107])
imagePoint2 = np.array([368.26071167, 224.86262512])

# Triangulate
point3D = cv2.triangulatePoints(P1, P2, imagePoint1, imagePoint2).T
point3D = point3D[:, :3] / point3D[:, 3:4]
print(point3D)

# Reproject back into the two cameras
rvec1, _ = cv2.Rodrigues(R1)
rvec2, _ = cv2.Rodrigues(R2)

p1, _ = cv2.projectPoints(point3D, rvec1, t1, K, distCoeffs=None)
p2, _ = cv2.projectPoints(point3D, rvec2, t2, K, distCoeffs=None)

# measure difference between original image point and reporjected image point 

reprojection_error1 = np.linalg.norm(imagePoint1 - p1[0, :])
reprojection_error2 = np.linalg.norm(imagePoint2 - p2[0, :])

print(reprojection_error1, reprojection_error2)
