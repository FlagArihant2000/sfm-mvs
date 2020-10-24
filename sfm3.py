# INCREEMENTAL STRUCTURE FROM MOTION
# AUTHORS: ARIHANT GAUR AND SAURABH KEMEKAR
# ORGANIZATION: IvLabs, VNIT

import cv2
import numpy as np
import os

#K = np.array([[2393.952166119461, -3.410605131648481e-13, 932.3821770809047], [0, 2398.118540286656, 628.2649953288065], [0, 0, 1]])
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

t0 = np.zeros((3,1))
R0 = np.eye(3)
Rt0 = np.hstack((R0, t0))
P0 = np.matmul(K, Rt0)
Pref = P0

#print(X.shape)
#sys.exit()
cameras = [P0]


Xtot = np.zeros((1,3))
colorstot = np.zeros((1,3))

features = []
descriptors = []
matches = []
i = 0
densify = False
tot_images = len(images)
#thresh_features = 200
match_interval = 3 # tot_images
# Acquisition of all features as well as matching
while(i < tot_images):
	print("Image: ",i)
	if downscale == 2:
		img = cv2.pyrDown(cv2.imread(img_dir +'/'+ images[i]))
	else:
		img = cv2.imread(img_dir +'/'+ images[i])
	
	sift = cv2.xfeatures2d.SIFT_create()
	imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	kp, des = sift.detectAndCompute(imggray, None)

	features = features + [kp]
	descriptors = descriptors + [des]
	
	if i == 0:
		i = i + 1
		continue
	
	j = 1
	while(j < i + 1):
		bf = cv2.BFMatcher()
		match = bf.knnMatch(descriptors[i - j], des, k = 2)
		#matches = matches + [match]
		good = []
		for m,n in match:
			if m.distance < 0.8 * n.distance:
				good.append(m)
				
		pts0 = np.float32([features[i - j][m.queryIdx].pt for m in good])
		pts1 = np.float32([kp[m.trainIdx].pt for m in good])
		
		E, mask = cv2.findEssentialMat(pts0, pts1, K, method = cv2.RANSAC, prob = 0.999, threshold = 0.4, mask = None)
		
		pts0 = pts0[mask.ravel() == 1]
		pts1 = pts1[mask.ravel() == 1]
		
		
		print(j)
		j = j + 1
	"""if i + 1 >= match_interval:
		j = match_interval - 1
		while(j > 0):
			bf = cv2.BFMatcher()
			match = bf.knnMatch(descriptors[i - match_interval + j], des, k=2)
			matches = matches + [match]
			#print(j)
			j = j - 1"""
	
	i = i + 1


	

		
		
