# STRUCTURE FROM MOTION
# AUTHOR: ARIHANT GAUR

import numpy as np
import cv2
import open3d as o3d


def visualize(cloud):
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(cloud)
	o3d.io.write_point_cloud('pc.ply',pcd)
	return None

# Intrinsic Camera Matrix
#K = np.array([[538.731, 0, 503.622],[0, 538.615, 265.447],[0, 0, 1]], dtype = np.float32)
K = np.array([[711.926, 0, 399.418],[0, 710.904, 243.435],[0, 0, 1]], dtype = np.float32)
# Can add CLAHE to increase the number of features 
clahe = cv2.createCLAHE(clipLimit = 5.0)

# First projection matrix
P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0]])

# SIFT feature detector
sift = cv2.xfeatures2d.SIFT_create()

# FLANN matcher parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 100)
flann = cv2.FlannBasedMatcher(index_params, search_params)

i = 1
final_cloud = np.array([])


while(i < 237):
	img1 = cv2.imread('/home/arihant/Desktop/sfm/delivery_area_rig_undistorted/delivery_area/images/images_rig_cam4_undistorted/'+str(i)+'.png')
	img2 = cv2.imread('/home/arihant/Desktop/sfm/delivery_area_rig_undistorted/delivery_area/images/images_rig_cam4_undistorted/'+str(i+1)+'.png')
	img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	
	kp1, des1 = sift.detectAndCompute(img1gray, None)
	kp2, des2 = sift.detectAndCompute(img2gray, None)
	#kp1 = np.array([kp1[idx].pt for idx in range(len(kp1))], dtype = np.float32)
	#kp2 = np.array([kp2[idx].pt for idx in range(len(kp2))], dtype = np.float32)
	
	matches = flann.knnMatch(des1, des2, k = 2)
	good = []
	pts1 = []
	pts2 = []
	for counter, (m,n) in enumerate(matches):
		if m.distance < 0.7 * n.distance:
			good.append(m)
			pts1.append(kp1[m.queryIdx].pt)
			pts2.append(kp2[m.trainIdx].pt)
	
	pts1 = np.float32(pts1)
	pts2 = np.float32(pts2)

	E, mask = cv2.findEssentialMat(pts1, pts2, K, method = cv2.RANSAC, prob = 0.999, threshold = 0.4, mask = None)
	pts1 = pts1[mask.ravel() == 1]
	pts2 = pts2[mask.ravel() == 1]
	
	_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
	
	P2 = np.hstack((R,t))
	P2 = K.dot(P2)
	pts1 = np.transpose(pts1)
	pts2 = np.transpose(pts2)
	pts1 = pts1.reshape(2, -1)
	pts2 = pts2.reshape(2, -1)
	#print(pts1)
	cloud = cv2.triangulatePoints(P1, P2, pts1, pts2).reshape(-1, 4)[:, :3]
	#cloud = cloud / cloud[3]
	#final_cloud = final_cloud + [cloud]
	if i == 1:
		final_cloud = cloud
	else:
		final_cloud = np.concatenate((final_cloud, cloud), axis = 0)
	#print(final_cloud.shape)
	#x = cv2.triangulatePoints(P1, P2, pts1, pts2).reshape(-1, 4)[:, :3]
	#print(E)
	
	cv2.imshow('image', img1)
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
	P1 = np.copy(P2)
	i = i + 1
	
x = visualize(final_cloud)
	
