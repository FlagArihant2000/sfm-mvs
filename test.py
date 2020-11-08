import cv2
import numpy as np
import os
from scipy.optimize import least_squares
import copy
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt

def feat_to_tracks(kp, poses):
	#print(kp.shape, poses.shape)
	tot_corrs = poses.shape[1]
	i = 0
	while(i < tot_corrs):
		pose = poses[:,i]
		r, t = pose[0:3], pose[3:6]
		R, _ = cv2.Rodrigues(r)
		#print(kp.shape)
		kp_h = cv2.convertPointsToHomogeneous(kp)[:, 0, :]
		kp_h = kp_h - t.T
		Rinv = np.linalg.inv(R)
		
		kp_h = np.array([np.matmul(Rinv, kp_) for kp_ in kp_h])
		kp = cv2.convertPointsFromHomogeneous(kp_h)[:, 0, :]
		print(kp)
		i = i + 1
		
	
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

posearr = K.ravel()
R_t_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
R_t_1 = np.empty((3, 4))

P1 = np.matmul(K, R_t_0)
Pref = P1
P2 = np.empty((3, 4))

Xtot = np.zeros((1, 3))
colorstot = np.zeros((1, 3))

sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()

img_list = sorted(os.listdir(img_dir))
images = []
for img in img_list:
    if '.jpg' in img.lower() or '.png' in img.lower():
        images = images + [img]
i = 1
#print(images)

# Acquiring the first image and detecting features using SIFT
img0 = img_downscale(cv2.imread(img_dir + '/' + images[0]), downscale)
img0gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
kp0, des0 = sift.detectAndCompute(img0gray, None)

img_tot = 3 #len(images)
feature_thresh = 20
all_poses = np.array([])
while(i < img_tot):
	
	img1 = img_downscale(cv2.imread(img_dir + '/' + images[i]), downscale)
	img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	kp1, des1 = sift.detectAndCompute(img1gray, None)
	
	matches = bf.knnMatch(des0, des1, k = 2)
	good = []
	for m, n in matches:
		if m.distance < 0.7 * n.distance:
			good.append(m)
	if i == 1:
		kp0 = np.float32([kp0[m.queryIdx].pt for m in good])
	else:
		kp0 = np.float32([kp0[m.queryIdx] for m in good])
	
	
	kp1 = np.float32([kp1[m.trainIdx].pt for m in good])
	des0 = np.float32([des0[m.queryIdx] for m in good])
	des1 = np.float32([des1[m.trainIdx] for m in good])
	
	E, mask = cv2.findEssentialMat(kp0, kp1, K, method=cv2.RANSAC, prob = 0.999, threshold = 1, mask = None)
	
	kp0 = kp0[mask.ravel() == 1]
	kp1 = kp1[mask.ravel() == 1]
	
	des0 = des0[mask.ravel() == 1]
	des1 = des1[mask.ravel() == 1]
	#if i != 1:
	#	find_common(kp1o, kp0, des1o, des0)
	
	_, R, t, mask = cv2.recoverPose(E, kp0, kp1, K)
	r, _ = cv2.Rodrigues(R)
	#print(r.shape, t.shape)
	Rt = np.vstack((r,t))
	if i == 1:
		all_poses = Rt
	else:
		all_poses = np.hstack((Rt, all_poses))
	"""if len(kp0) < feature_thresh:
		print("Frame: ",i, "Less features! Restart tracks")
		#print("Frame: ",i,",Total features tracked: ",len(kp0))
	else:
		print("Frame: ",i,",Total features tracked: ",len(kp0))"""
	
	kp0 = kp1
	kp1o = kp1
	des1o = des1
	img0 = img1
	des0 = des1
	img0gray = img1gray
	cv2.imshow('image', img1)
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
	i = i + 1
#print(des0)
feat_to_tracks(kp1, all_poses)
cv2.destroyAllWindows()
