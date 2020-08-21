import cv2
import numpy as np
import os

img_directory = '/home/arihant/structure-from-motion/'

totlist = sorted(os.listdir(img_directory))
imglist = []
for x in totlist:
	if '.jpg' in x:
		imglist = imglist + [x]
#print(imglist)
imgenum = 0
#K = np.array([[1520.400000, 0.000000, 302.320000], [0.000000, 1525.900000, 246.870000], [0.000000, 0.000000, 1.000000]])
K = np.array([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]])
downscale = 2

K[0,0] = K[0,0] / float(downscale)
K[1,1] = K[1,1] / float(downscale)
K[0,2] = K[0,2] / float(downscale)
K[1,2] = K[1,2] / float(downscale)
D = np.zeros((5,1), dtype = np.float32)

sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()
R_ref = np.eye(3)
t_ref = np.array([[0],[0],[0]], dtype = np.float32)

max_disparity = 199
min_disparity = 23
num_disparities = max_disparity - min_disparity
window_size = 5
stereo = cv2.StereoSGBM_create(minDisparity = min_disparity, numDisparities = num_disparities, blockSize = 5, uniquenessRatio = 2, speckleWindowSize = 15, speckleRange = 5, disp12MaxDiff = 2, P1 = 8*3*window_size**2, P2 = 32*3*window_size**2)

stereo2 = cv2.ximgproc.createRightMatcher(stereo)

lamb = 8000
sig = 1.5
visual_multiplier = 1.0
wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
wls_filter.setLambda(lamb)
wls_filter.setSigmaColor(sig)
#print(imglist)

img1 = cv2.pyrDown(cv2.imread(img_directory + imglist[imgenum + 0]))
img2 = cv2.pyrDown(cv2.imread(img_directory + imglist[imgenum + 1]))
img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#print(img1.shape)

kp1, des1 = sift.detectAndCompute(img1gray, None)
kp2, des2 = sift.detectAndCompute(img2gray, None)

matches = bf.knnMatch(des1, des2, k = 2)

good = []
for m,n in matches:
	if m.distance < 0.70 * n.distance:
		good.append(m)
		
pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

E, mask = cv2.findEssentialMat(pts1, pts2, K, method = cv2.RANSAC, prob = 0.999, threshold = 0.4, mask = None)
#print(E)
cv2.namedWindow('WLS Filtered', cv2.WINDOW_NORMAL)
cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
print(img1.shape)
if E is not None:
	pts1 = pts1[mask.ravel() ==1]
	pts2 = pts2[mask.ravel() ==1]
	_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

	P1 = np.hstack((R_ref, t_ref))
	P1 = np.matmul(K, P1)
	
	R_ref = R_ref.dot(R)
	t_ref = t_ref + R_ref.dot(t)
	
	P2 = np.hstack((R_ref, t_ref))
	P2 = np.matmul(K, P2)
	
	points1 = pts1.reshape(2, -1)
	points2 = pts2.reshape(2, -1)
	print(P1, P2)
	R1, R2, P1, P2, Q, a, b = cv2.stereoRectify(K, D, K, D, (1536, 1024), R, t)
	map1, map2 = cv2.initUndistortRectifyMap(K, D, R1, P1, (1536, 1024), cv2.CV_16SC2)
	img1rec = cv2.remap(img1, map1, map2, cv2.INTER_CUBIC)

	map3, map4 = cv2.initUndistortRectifyMap(K, D, R2, P2, (1536, 1024), cv2.CV_16SC2)
	img2rec = cv2.remap(img2, map3, map4, cv2.INTER_CUBIC)
	#img1rec, img2rec = img1, img2
	disparity = stereo.compute(img1rec, img2rec)
	disparity = np.int16(disparity)
	_, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)
	#disparity = (disparity / 16).astype(np.uint8)
	
	disparity2 = stereo2.compute(img2rec, img1rec)
	disparity2 = np.int16(disparity2)
	_, disparity2 = cv2.threshold(disparity2, 0, max_disparity * 16, cv2.THRESH_TOZERO)
	#disparity2 = (disparity2 / 16).astype(np.uint8)
	
	#disparity = np.int16(disparity)
	#_, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)
	#disparity = (disparity / 16).astype(np.uint8)
	
	
	
	filteredImg = wls_filter.filter(disparity, img2, None, disparity2)
	_, filteredImg = cv2.threshold(filteredImg, 0, max_disparity * 16, cv2.THRESH_TOZERO)
	filteredImg = (filteredImg / 16).astype(np.uint8)

			
	
	cv2.imshow('image1', img1rec)
	cv2.imshow('image2', img2rec)
	cv2.imshow('WLS Filtered', filteredImg)
	#cv2.imshow('disparity', disparity)
		
cv2.waitKey(0)
cv2.destroyAllWindows()

