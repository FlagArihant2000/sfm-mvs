# STRUCTURE FROM MOTION 
# Author: Arihant Gaur
# Organization: IvLabs, VNIT

import cv2
import numpy as np
import os


def Triangulation(P1, R, t, pts1, pts2, K):
	
	P2 = np.hstack((R, t))
	P2 = np.matmul(K, P2)
	
	points1 = pts1.reshape(2, -1)
	points2 = pts2.reshape(2, -1)
	
	cloud = cv2.triangulatePoints(P1, P2, points1, points2)
	cloud = cloud / cloud[3]
	cloud = cv2.convertPointsFromHomogeneous(cloud.T)
	
	return cloud, P2
	
def ReprojectionError(X, pts, R, t, K):
	total_error = 0
	
	r, _ = cv2.Rodrigues(R)
	
	#print(X.shape)
	
	p, _ = cv2.projectPoints(X, r, t, K, distCoeffs = None)
	
	p = p[:, 0, :]
	
	error = cv2.norm(p, pts, cv2.NORM_L2)/len(p)
	
	tot_error = error ** 2
	tot_error = tot_error / len(X)
	
	return tot_error
	
def PnP(X, p2, K, dist_coeff):

	ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(X, p2, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)
	
	R, J = cv2.Rodrigues(rvecs)

	return R, tvecs

def to_ply(point_cloud, colors):
	#colors = np.zeros_like(point_cloud)
	#kp_new = np.array([kp[i][0] for i in range(len(kp))])
	#kp_new = np.array(kp_new, dtype = np.int32)
	#kp_new = kp
	#colors = np.array([img2[l[1],l[0]] for l in kp_new])

	#out_points = points[mask]
	#out_colors = image[mask]
	
	#out_points = kp_new.reshape(-1,3)
	out_points = point_cloud.reshape(-1,3)
	out_colors = colors.reshape(-1,3)
	verts = np.hstack([out_points, out_colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar blue
		property uchar green
		property uchar red
		end_header
		'''
	with open('stereo.ply', 'w') as f:
		f.write(ply_header %dict(vert_num = len(verts)))
		np.savetxt(f, verts, '%f %f %f %d %d %d')

# Locating Images
img_directory = '/home/arihant/structure-from-motion/'

files = sorted(os.listdir(img_directory))
#files = sorted(os.listdir(img_directory))
images = []
for im in files:
	if '.jpg' in im or '.png' in im:
		images = images + [im]

# Defining Camera Parameters		
K = np.array([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]])
D = np.zeros((5,1), dtype = np.float32)

# Defining Downscaling factor to reduce computational power requires
downscale = 2

K[0,0] = K[0,0] / float(downscale)
K[1,1] = K[1,1] / float(downscale)
K[0,2] = K[0,2] / float(downscale)
K[1,2] = K[1,2] / float(downscale)

# Feature Detection and Matching Initialization (SIFT and Brute Force KNN)
sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()


i = 0 # Counter Initialization

# Window Initialization
cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)

Rtot = np.eye(3)
ttot = np.zeros((3,1))

while(i < len(images) - 1):

	# Image Acquisition
	if downscale == 2:
		img1 = cv2.pyrDown(cv2.imread(img_directory + images[i]))
		img2 = cv2.pyrDown(cv2.imread(img_directory + images[i + 1]))
	else:
		img1 = cv2.imread(img_directory + images[i])
		img2 = cv2.imread(img_directory + images[i + 1])
		
	img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	
	# Feature Detection
	kp1, des1 = sift.detectAndCompute(img1gray, None)
	kp2, des2 = sift.detectAndCompute(img2gray, None)
	
	# Feature Matching and Acquiring best matches. Lowe's Ratio = 0.7
	matches = bf.knnMatch(des1, des2, k = 2)
	good = []
	for m,n in matches:
		if m.distance < 0.70 * n.distance:
			good.append(m)
			
	pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
	pts2 = np.float32([kp2[m.trainIdx].pt for m in good])	
	
	# Finding Essential Matrix
	E, mask = cv2.findEssentialMat(pts1, pts2, K, method = cv2.RANSAC, prob = 0.999, threshold = 0.4, mask = None)

	# Outlier Rejection
	pts1 = pts1[mask.ravel() == 1]
	pts2 = pts2[mask.ravel() == 1]
	
	# Pose Recovery of the second camera
	_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
	if i == 0:
		P1 = np.hstack((np.eye(3), np.zeros((3,1))))
	else:
		#P1 = [Rtot, ttot]
		P1 = np.hstack((np.eye(3), np.zeros((3,1))))
	
	X, P2 = Triangulation(P1, R, t, pts1, pts2, K)
	error = ReprojectionError(X, pts1, R, t, K)
	Rnew, tnew = PnP(X, pts2, K, D)
	Rtot = np.matmul(Rtot, Rnew)
	ttot = ttot + np.matmul(Rtot, tnew)
	#error = ReprojectionError(X, pts1, Rnew, tnew, K)
	print(error)
	
	# Displaying Images
	cv2.imshow('image1', img1)
	cv2.imshow('image2', img2)
	i = i + 1
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
		
cv2.destroyAllWindows()
	
	
	
