import cv2
import numpy as np
import os
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import copy
import open3d as o3d

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

def to_ply(path,point_cloud, colors, densify):
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
	if not densify:
		with open(path+'/Point_Cloud/sparse.ply', 'w') as f:
			f.write(ply_header %dict(vert_num = len(verts)))
			np.savetxt(f, verts, '%f %f %f %d %d %d')
	else:
		with open(path+'/Point_Cloud/dense.ply', 'w') as f:
			f.write(ply_header %dict(vert_num = len(verts)))
			np.savetxt(f, verts, '%f %f %f %d %d %d')

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

'''posefile = open(img_dir+'/poses.txt','w')
posefile.write("K = " + str(K.flatten()).replace('\n',''))
posefile.write("\n")
posefile.write(str(i) + " = " + str(R_t_0.flatten()).replace('\n',''))
posefile.write("\n")
fpfile = open(img_dir+'/features.txt','w')'''

img0 = cv2.pyrDown(cv2.imread(img_dir +'/'+ images[i]))

keypoints = []
descriptors = []
matches = []
#frame_ind = []
imgs = []
Xtot = np.zeros((1,3))
colorstot = np.zeros((1,3))

visualize_first_cloud = True

print("Detecting Features...")
# FEATURE DETECTION
for i, im in enumerate(images):
	if downscale == 2:
		img = cv2.pyrDown(cv2.imread(img_dir + '/' + im))
	else:
		img = cv2.imread(img_dir + '/' + im)
	
	imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(imggray, None)
	keypoints = keypoints + [kp]
	descriptors = descriptors + [des]
	imgs = imgs + [img]
	#print(i)
print("Matching features for image pairs...")
points0 = []
points1 = []
for c in range(len(images) - 1):
	#print(c)
	des0 = descriptors[c]
	des1 = descriptors[c + 1]
	kp0 = keypoints[c]
	kp1 = keypoints[c + 1]
	
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des0, des1, k = 2)
	
	good = []
	for m,n in matches:
		if m.distance < 0.70 * n.distance:
			good.append(m)
	
	pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
	pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
	points0 = points0 + [pts0]
	points1 = points1 + [pts1]

keypoints = []	
print("First Pair Essential Matrix...")
E, mask = cv2.findEssentialMat(points0[0], points1[0], K, method = cv2.RANSAC, prob = 0.9, threshold = 5.0, mask = None)
points0[0] = points0[0][mask.ravel() == 1]
points1[0] = points1[0][mask.ravel() == 1]
print("Pose Recovery...")
_, R, t, mask = cv2.recoverPose(E, points0[0], points1[0], K)
	
points0[0] = points0[0][mask.ravel() > 0]
points1[0] = points1[0][mask.ravel() > 0]
#points0[1] = points1[0]
print("Triangulation for first pair...")
X = Triangulation(R, t, K, points0[0], points1[0])
#print(X.shape)

if visualize_first_cloud:
	Xtot = np.vstack((Xtot, X[:, 0, :]))
	pts1_reg = np.array(points1[0], dtype = np.int32)
	colors = np.array([imgs[1][l[1],l[0]] for l in pts1_reg])
	colorstot = np.vstack((colorstot, colors))

densify = False
to_ply(path, Xtot, colorstot, densify)
cv2.destroyAllWindows()



