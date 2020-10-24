import cv2
import numpy as np
import os

def pair_match(des0, des1, kp0, kp1):
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des0, des1, k = 2)

	good = []
	for m,n in matches:
		if m.distance < 0.70 * n.distance:
			good.append(m)
	pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
	pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
	return pts0, pts1
	
# Match array order 01 02 12 03 13 23 .... where the corresponding digit represents index for first and second image
def match_imgs(i, interval, tot, features, descriptors):
	initial = tot[i + 1 - interval: i + 1]
	prev = tot[0 : i + 1  - interval]
	initial = np.array(initial)
	prev = np.array(prev)
	prev = np.sum(prev)

	j = 0
	temp = 0
	features_arr = []
	descriptors_arr = []
	left_image = []
	right_image = []
	while (j < interval):
		kp = features[int(prev) + temp: initial[j] + int(prev) + temp]
		des = descriptors[int(prev) + temp: initial[j] + int(prev) + temp]
		temp = temp + initial[j]
		if j >= 1:
			k = 0
			while(k < j):
				pts0, pts1 = pair_match(descriptors_arr[k], des, features_arr[k], kp)
				left_image = left_image + [pts0]
				right_image = right_image + [pts1]
				k = k + 1
				
		features_arr = features_arr + [kp]
		descriptors_arr = descriptors_arr + [des]
		#print(len(left_image), len(right_image))
		j = j + 1
	return left_image, right_image
	
def Triangulation(P1, P2, pts1, pts2, K):


	points1 = np.transpose(pts1)
	points2 = np.transpose(pts2)

	cloud = cv2.triangulatePoints(P1, P2, points1, points2)
	
	#cloud = cloud / cloud[3]
	
	cloud = cv2.convertPointsFromHomogeneous(cloud.T)

	return cloud
	
def ImageRegistration(X, pts0, pts1, K):

	X = X[:, 0, :]

	d = np.zeros((5,1))
	print(X.shape, pts1.shape)
	ret, rvecs, t, inliers = cv2.solvePnPRansac(X, pts1, K, d, cv2.SOLVEPNP_ITERATIVE)
	R, _ = cv2.Rodrigues(rvecs)

	if inliers is not None:
		pts0 = pts0[inliers[:,0]]
		X = X[inliers[:,0]]
		pts1 = pts1[inliers[:,0]]

	return R, t, X, pts0, pts1

def setPointCloud(start_idx, left, right, K, color):
	j = 0
	maxfeatures, idx = 0, 0
	essentials = []
	while(j < len(left)):
		pts0 = left[j]
		pts1 = right[j]
		if len(pts0) > maxfeatures:
			maxfeatures = len(pts0)
			idx = j
		
		E, mask = cv2.findEssentialMat(pts0, pts1, K, method = cv2.RANSAC, prob = 0.999, threshold = 0.4, mask = None)
		essentials = essentials + [E]
		pts0 = pts0[mask.ravel() == 1]
		pts1 = pts1[mask.ravel() == 1]
		left[j] = pts0
		right[j] = pts1
		j = j + 1
	
	Rt0 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
	P0 = np.matmul(K, Rt0)
	
	_ , R, t, mask = cv2.recoverPose(essentials[idx], left[idx], right[idx], K)
	left[idx] = left[idx][mask.ravel() > 0]
	right[idx] = right[idx][mask.ravel() > 0]
	
	Rt = np.hstack((R,t))
	P1 = np.matmul(K, Rt)
	#print(idx)
	# Set the base for point cloud
	X = Triangulation(P0, P1, left[idx], right[idx], K)
	
	# Register the remaining images
	R, t, X, left[idx], right[idx] = ImageRegistration(X, left[idx], right[idx], K)
	
	repr_error = ReprojectionError(X, right[idx], R, t, K)
	print("Reprojection Error: ",repr_error)
	
def ReprojectionError(X, pts, R, t, K):
	total_error = 0

	r, _ = cv2.Rodrigues(R)

	p, _ = cv2.projectPoints(X, r, t, K, distCoeffs = None)
	p = p[:, 0, :]
	p = np.float32(p)
	pts = np.float32(pts)

	total_error = cv2.norm(p, pts, cv2.NORM_L2)
	pts = pts.T
	tot_error = total_error/len(p)
	#print(p, pts.T)

	return tot_error

# Structure from Motion (Pipeline is related to COLMAP)
# Assumption: We are familiar with image sequence, as per their naming.

K = np.array([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]])
downscale = 2

K[0,0] = K[0,0] / float(downscale)
K[1,1] = K[1,1] / float(downscale)
K[0,2] = K[0,2] / float(downscale)
K[1,2] = K[1,2] / float(downscale)

cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)

path = os.getcwd()
#img_dir = path + '/Dataset2/'
img_dir = path + '/Dataset/'

#img_dir = '/home/arihant/Desktop/uoft/'
img_list = sorted(os.listdir(img_dir))

images = []

for img in img_list:
	if '.JPG' in img or '.jpg' in img:
		images = images + [img]


descriptors = np.array([])
features = np.array([])
matches = []
tot_des = []
i = 0
tot_images = len(images)
tot_match_imgs = 3 # Optimum number is 3; can be scaled as large as the total length of dataset
color = True

# ACQUIRE FEATURES FOR ALL IMAGES. ALSO PERFORM MATCHING FOR tot_match_imgs QUANTITY AT A TIME
while(i < tot_images):
	print("Image: ",i)
	if downscale == 2:
		img = cv2.pyrDown(cv2.imread(img_dir +'/'+ images[i]))
	else:
		img = cv2.imread(img_dir +'/'+ images[i])
		
	sift = cv2.xfeatures2d.SIFT_create()
	imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	kp, des = sift.detectAndCompute(imggray, None)
	if i == 0:
		features = np.array(kp)
		descriptors = np.array(des)
	else:
		features = np.append(features, kp)
		descriptors = np.vstack((descriptors, des))
		
	tot_des = tot_des + [len(des)]
	#print(features.shape, descriptors.shape)
	
	if i + 1 >= tot_match_imgs:
		left, right = match_imgs(i, tot_match_imgs, tot_des, features, descriptors)
		print(len(left), len(right))
		
	if i + 1 == tot_match_imgs:
		setPointCloud(i, left, right, K, color)
	#else:
	#	updatePointCloud()
	
	i = i + 1


	



