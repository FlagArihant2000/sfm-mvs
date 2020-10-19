import cv2
import numpy as np
import os
import sys

def Triangulation(P0, P1, pts0, pts1):
	
	#Rt0 = np.hstack((R0, t0))
	#Rt1 = np.hstack((R,t))
	
	#P0 = np.matmul(K, Rt0)
	#P1 = np.matmul(K, Rt1)
	
	P0 = np.float32(P0)
	P1 = np.float32(P1)
	
	pts0 = np.float32(pts0.T)
	pts1 = np.float32(pts1.T)
	X = cv2.triangulatePoints(P0, P1, pts0, pts1)
	X = cv2.convertPointsFromHomogeneous(X.T)
	return np.array(X)

def PnP(p0, p, X, K, P):
	#X = X[:, 0, :]
	#p = p.T
	#p0 = p0.T
	# Find the corresponding 3D points with the new image parameters. We have projection matrix for previous image and corresponding new image points. From this, find newer 3D points.
	
	
	d = np.zeros((5,1))
	print(X.shape, p.shape, K.shape, d.shape)
	ret, rvecs, t, inliers = cv2.solvePnPRansac(X, p, K, d, cv2.SOLVEPNP_ITERATIVE)
	
	R, _ = cv2.Rodrigues(rvecs)
	
	if inliers is not None:
		p = p[inliers[:,0]]
		X = X[inliers[:,0]]
		#p0 = p0[inliers[:,0]]

	return R, t, p, p0, X
	
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

K = np.array([[2393.952166119461, -3.410605131648481e-13, 932.3821770809047], [0, 2398.118540286656, 628.2649953288065], [0, 0, 1]])
#K = np.array([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]])
downscale = 2

K[0,0] = K[0,0] / float(downscale)
K[1,1] = K[1,1] / float(downscale)
K[0,2] = K[0,2] / float(downscale)
K[1,2] = K[1,2] / float(downscale)

cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)

path = os.getcwd()
#img_dir = path + '/Dataset2/'
#img_dir = '/home/arihant/sfm-mvs/Dataset/'

img_dir = '/home/arihant/Desktop/uoft/'
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


i = 0
densify = False
while(i < len(images) - 1):

	#print("Image Acquisition...")
	if downscale == 2:
		img0 = cv2.pyrDown(cv2.imread(img_dir +'/'+ images[i]))
		img1 = cv2.pyrDown(cv2.imread(img_dir +'/'+ images[i + 1]))
	else:
		img0 = cv2.imread(img_dir +'/'+ images[i])
		img1 = cv2.imread(img_dir +'/'+ images[i + 1])
	
	img0gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
	img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	
	#print("Feature Detection...")
	sift = cv2.xfeatures2d.SIFT_create()
	kp0, des0 = sift.detectAndCompute(img0gray, None)
	kp1, des1 = sift.detectAndCompute(img1gray, None)
	
	#print("Feature Matching...")
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des0, des1, k=2)
	good = []
	for m,n in matches:
		if m.distance < 0.8 * n.distance:
			good.append(m)
	
	pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
	pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
	
	#print("Essential Matrix and Pose Recovery...")
	E, mask = cv2.findEssentialMat(pts0, pts1, K, method = cv2.RANSAC, prob = 0.999, threshold = 0.4, mask = None)

	pts0 = pts0[mask.ravel() == 1]
	pts1 = pts1[mask.ravel() == 1]
	_, R, t, mask = cv2.recoverPose(E, pts0, pts1, K)
	
	pts0 = pts0[mask.ravel() > 0]
	pts1 = pts1[mask.ravel() > 0]
	
	Rt = np.hstack((R, t))
	P = np.matmul(K, Rt)
	
	X = Triangulation(Pref, P, pts0, pts1)
	
	
	Rpnp, tpnp, pts0, pts1, X = PnP(pts0, pts1, X, K, P)
	
	Rtpnp = np.hstack((Rpnp, tpnp))
	Ppnp = np.matmul(K, Rtpnp)
	
	R1 = np.matmul(R0, Rpnp)
	t1 = t0 + np.matmul(R1, tpnp)
	
	
	Rt1 = np.hstack((R1,t1))
	P1 = np.matmul(K, Rt1)
	
	cameras = cameras + [P1]
	#print(P0, P1)
	points_3d = Triangulation(P0, P1, pts0, pts1)
	
	Xtot = np.vstack((Xtot, points_3d[:, 0, :]))
	pts1_reg = np.array(pts1, dtype = np.int32)
	colors = np.array([img1[l[1],l[0]] for l in pts1_reg])
	colorstot = np.vstack((colorstot, colors))
	
	cv2.imshow('image1', img0)
	cv2.imshow('image2', img1)
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
	
	print(i)
	R0 = R1
	t0 = t1
	P0 = P1
	i = i + 1
	#break
	
	
print("Processing Point Cloud...")
to_ply(path, Xtot, colorstot, densify)
cv2.destroyAllWindows()

print("DONE")

