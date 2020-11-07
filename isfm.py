import cv2
import numpy as np
import os
from scipy.optimize import least_squares
import copy
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt


def img_downscale(img, downscale):
	downscale = int(downscale/2)
	i = 1
	while(i <= downscale):
		img = cv2.pyrDown(img)
		i = i + 1
	return img


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# Input Camera Intrinsic Parameters
K = np.array([[2393.952166119461, -3.410605131648481e-13, 932.3821770809047], [0, 2398.118540286656, 628.2649953288065], [0, 0, 1]])

# Suppose if computationally heavy, then the images can be downsampled once. Note that downsampling is done in powers of two, that is, 1,2,4,8,...
downscale = 2
K[0,0] = K[0,0] / float(downscale)
K[1,1] = K[1,1] / float(downscale)
K[0,2] = K[0,2] / float(downscale)
K[1,2] = K[1,2] / float(downscale)

# Current Path Directory
path = os.getcwd()

# Input the directory where the images are kept. Note that the images have to be named in order for this particular implementation
#img_dir = path + '/Sample Dataset/'
img_dir = '/home/arihant/Desktop/gustav/'

img_list = sorted(os.listdir(img_dir))
images = []
for img in img_list:
	if '.jpg' in img.lower() or '.png' in img.lower():
		images = images + [img]

#print(images)
i = 0
sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher(crossCheck = False)

indexes = np.array([], dtype = np.int16)
tot_images = 10
descriptors = np.array([])
keypoints = np.array([])



while(i < tot_images):
	img = img_downscale(cv2.imread(img_dir + '/' + images[i]), downscale)
	imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	kp, des = sift.detectAndCompute(imggray, None)
	
	if i == 0:
		descriptors = np.array(des)
	else:
		descriptors = np.vstack((descriptors, des))
	#if i != 0:
	j = 0
	while(j < i):
		kp0 = keypoints[indexes[0:j].sum():indexes[0:j + 1].sum()]
		des0 = descriptors[indexes[0:j].sum():indexes[0:j + 1].sum()]
		matches = bf.knnMatch(des0, des, k=2)
		
		good = []
		for m, n in matches:
			if m.distance < 0.7 * n.distance:
				good.append(m)
		pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
		pts1 = np.float32([kp[m.trainIdx].pt for m in good])
		
		E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob = 0.999, threshold = 0.4, mask = None)
		pts0 = pts0[mask.ravel() == 1]
		pts1 = pts1[mask.ravel() == 1]
		_, R, t, mask = cv2.recoverPose(E, pts0, pts1, K)
		pts0 = pts0[mask.ravel() > 0]
		pts1 = pts1[mask.ravel() > 0]
		print(len(pts0))
		j = j + 1
	keypoints = np.append(keypoints, kp)
	
	indexes = np.append(indexes, len(kp))
	cv2.imshow('image', img)
	i = i + 1
	if cv2.waitKey(1) & 0xff == ord('q'):
		break

i = 0


cv2.destroyAllWindows()





