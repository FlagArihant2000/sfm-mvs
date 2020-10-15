import cv2
import numpy as np
import os
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import copy
import open3d as o3d
from tqdm import tqdm

def camera_orientation(path,mesh,R_T,i):
	T = np.zeros((4,4))
	T[:3,] = R_T
	T[3,:] = np.array([0,0,0,1])
	new_mesh = copy.deepcopy(mesh).transform(T)
	#print(new_mesh)
	#new_mesh.scale(0.5, center=new_mesh.get_center())
	o3d.io.write_triangle_mesh(path+"/Point_Cloud/camerapose"+str(i)+'.ply', new_mesh)

	return
	
def Triangulation(R, t, K, pts0, pts1):
	R0 = np.eye(3)
	t0 = np.zeros((3,1))
	Rt0 = np.hstack((R0, t0))
	Rt1 = np.hstack((R,t))
	
	P0 = np.matmul(K, Rt0)
	P1 = np.matmul(K, Rt1)
	
	P0 = np.float32(P0)
	P1 = np.float32(P1)
	
	pts0 = np.float32(pts0.T)
	pts1 = np.float32(pts1.T)
	
	X = cv2.triangulatePoints(P0, P1, pts0, pts1)
	X = cv2.convertPointsFromHomogeneous(X.T)
	return np.array(X)

def PnP(p0, p, X, K):
	X = X[:, 0, :]
	#p = p.T
	#p0 = p0.T
	d = np.zeros((5,1))
	ret, rvecs, t, inliers = cv2.solvePnPRansac(X, p, K, d, cv2.SOLVEPNP_ITERATIVE)
	
	R, _ = cv2.Rodrigues(rvecs)
	
	if inliers is not None:
		p = p[inliers[:,0]]
		X = X[inliers[:,0]]
		p0 = p0[inliers[:,0]]

	return R, t, p, p0, X

cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)

K = np.array([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]])
#K = np.array([[1520.400000, 0.000000, 302.320000], [0.000000, 1525.900000, 246.870000], [0.000000, 0.000000, 1.000000]])
#K = np.array([[2393.952166119461, -3.410605131648481e-13, 932.3821770809047], [0, 2398.118540286656, 628.2649953288065], [0, 0, 1]])
downscale = 2

K[0,0] = K[0,0] / float(downscale)
K[1,1] = K[1,1] / float(downscale)
K[0,2] = K[0,2] / float(downscale)
K[1,2] = K[1,2] / float(downscale)


R_t_0 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
R_t_1 = np.empty((3,4))

P1 = np.matmul(K, R_t_0)
Pref = P1
P2 = np.empty((3,4))


Xtot = np.zeros((1,3))
colorstot = np.zeros((1,3))

path = os.getcwd()
img_dir = path + '/Dataset'
#img_dir = path + '/Dataset2'
#img_dir = '/home/arihant/Desktop/SfM_quality_evaluation-master/Benchmarking_Camera_Calibration_2008/castle-P30/images/'
# Other Directories: fountain-P11, castle-P19, castle-P30, entry-P10, Herz-Jesus-P25

img_list = sorted(os.listdir(img_dir))
images = []
for img in img_list:
	if '.jpg' in img or '.jpeg' in img:
		images = images + [img]
i = 0
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

#camera_orientation(path,mesh,R_t_0,i)

posefile = open(img_dir+'/poses.txt','w')
posefile.write("K = " + str(K.flatten()).replace('\n',''))
posefile.write("\n")
posefile.write(str(i) + " = " + str(R_t_0.flatten()).replace('\n',''))
posefile.write("\n")
fpfile = open(img_dir+'/features.txt','w')

apply_ba = False
densify = False # Application of Patch based MVS to make a denser point cloud
#images = images[3:7]


img0 = cv2.pyrDown(cv2.imread(img_dir +'/'+ images[i]))
img0gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp0, des0 = sift.detectAndCompute(img0gray, None)
for i in tqdm(range(len(images)-1)):
	print(img_dir +'/'+ images[i])
	# IMAGE ACQUISITION
	img1 = cv2.pyrDown(cv2.imread(img_dir +'/'+ images[i + 1]))
	
	img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	
	# FEATURE DETECTION
	sift = cv2.xfeatures2d.SIFT_create()
	kp1, des1 = sift.detectAndCompute(img1gray, None)
	
	# FEATURE MATCHING
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des0, des1, k = 2)
	
	# Lowe's Test
	good = []
	for m,n in matches:
		if m.distance < 0.70 * n.distance:
			good.append(m)

	pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
	pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
	
	# ESSENTIAL MATRIX CALCULATION
	E, mask = cv2.findEssentialMat(pts0, pts1, K, method = cv2.RANSAC, prob = 0.9, threshold = 0.5, mask = None)

	pts0 = pts0[mask.ravel() == 1]
	pts1 = pts1[mask.ravel() == 1]
	
	# POSE RECOVERY
	_, R, t, mask = cv2.recoverPose(E, pts0, pts1, K)
	
	pts0 = pts0[mask.ravel() > 0]
	pts1 = pts1[mask.ravel() > 0]
	
	# LINEAR TRIANGULATION
	X = Triangulation(R, t, K, pts0, pts1)
	
	cv2.imshow('image1', img0)
	cv2.imshow('image2', img1)
	
	# PERSPECTIVE - N - POINT 
	R, t, pts0, pts1, X = PnP(pts0, pts1, X, K)
	
	img0 = img1
	img0gray = img1gray
	kp0 = kp1
	des0 = des1
	
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
		
cv2.destroyAllWindows()







