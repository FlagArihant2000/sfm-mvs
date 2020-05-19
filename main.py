# STRUCTURE FROM MOTION
# AUTHOR: ARIHANT GAUR


import cv2
import numpy as np

K = np.array([[538.731, 0, 503.622],[0, 538.615, 265.447],[0, 0, 1]], dtype = np.float32)


i = 2
sift = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 1

img1 = cv2.imread('/home/arihant/Desktop/sfm/delivery_area_rig_undistorted/delivery_area/images/images_rig_cam4_undistorted/'+str(i)+'.png')
img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

R_t_0 = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0]])
R_t_t1=  np.empty((3,4))
P1 = np.matmul(K, R_t_0)
#cloud = []
X = np.array([])
Y = np.array([])
Z = np.array([])

while(i <= 237):
	img2 = cv2.imread('/home/arihant/Desktop/sfm/delivery_area_rig_undistorted/delivery_area/images/images_rig_cam4_undistorted/'+str(i)+'.png')
	img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	kp1, des1 = sift.detectAndCompute(img1gray, None)
	kp2, des2 = sift.detectAndCompute(img2gray, None)
	
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 100)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k = 2)
	good = []
	pts1 = []
	pts2 = []
	for counter, (m, n) in enumerate(matches):
		if m.distance < 0.7 * n.distance:
			good.append(m)
			pts1.append(kp1[m.queryIdx].pt)
			pts2.append(kp2[m.trainIdx].pt)
	
	pts1 = np.int32(pts1)
	pts2 = np.int32(pts2)
	
	F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
	pts1 = pts1[mask.ravel() == 1]
	pts2 = pts2[mask.ravel() == 1]
	E = np.matmul(np.matmul(np.transpose(K), F), K)
	retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
	R_t_1 = np.matmul(R, R_t_0[:3,:3])
	R_t_1 = R_t_0[:3, 3] + np.matmul(R_t_0[:3,:3], t.ravel())
	P2 = np.matmul(K, R_t_1)
	pts1 = np.transpose(pts1)
	pts2 = np.transpose(pts2)
	cloud = cv2.triangulatePoints(P1, P2, pts1, pts2)
	cloud = cloud / cloud[3]
	print(cloud)
	cv2.imshow('image',img2)
	print(i)
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
	i = i + 1
	img1 = img2
	img1gray = img2gray

cv2.destroyAllWindows()
