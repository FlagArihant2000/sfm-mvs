# Structure From Motion for Image Pair

# AUTHOR: Arihant Gaur
# ORGANIZATION: IvLabs, VNIT

import cv2
import numpy as np


# Intrinsicn Matrix
K = np.array([[2780.1700000000000728, 0, 1539.25], [0, 2773.5399999999999636, 1001.2699999999999818], [0, 0, 1]])
D = np.zeros((5,1), dtype = np.float32)
downscale = 2
K[0,0] = K[0,0] / float(downscale)
K[1,1] = K[1,1] / float(downscale)
K[0,2] = K[0,2] / float(downscale)
K[1,2] = K[1,2] / float(downscale)

# For two images
imgL = cv2.pyrDown(cv2.imread('img1.ppm'))
imgR = cv2.pyrDown(cv2.imread('img2.ppm'))

imgLgray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgRgray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)


sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(imgLgray, None)
kp2, des2 = sift.detectAndCompute(imgRgray, None)

# Feature Matching and Outlier Rejection
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k = 2)

good = []
for m,n in matches:
	if m.distance < 0.70 * n.distance:
		good.append(m)
		
pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

E, mask = cv2.findEssentialMat(pts1, pts2, K, method = cv2.RANSAC, prob = 0.999, threshold = 0.4, mask = None)
pts1 = pts1[mask.ravel() ==1]
pts2 = pts2[mask.ravel() ==1]
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

P0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P0 = K.dot(P0)
P1 = np.hstack((R, t))
P1 = K.dot(P1)
points1 = kp1.reshape(2, -1)
points2 = kp2.reshape(2, -1)
cloud = cv2.triangulatePoints(P0, P1, points1, points2).reshape(-1, 4)[:, :3]

cv2.imshow('image Left', imgL)
cv2.waitKey(0)
cv2.destroyAllWindows()
