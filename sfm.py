# Structure from Motion
# Authors: Arihant Gaur and Saurabh Kemekar
# Organization: IvLabs, VNIT


import cv2
import numpy as np
import os
from scipy.optimize import least_squares
import copy
import open3d as o3d

def Triangulation(P1, P2, pts1, pts2, K, repeat):
	
	if not repeat:
		points1 = np.transpose(pts1)
		points2 = np.transpose(pts2)
	else:
		points1 = pts1
		points2 = pts2
	
	cloud = cv2.triangulatePoints(P1, P2, points1, points2)
	cloud = cloud / cloud[3]

	return points1, points2, cloud
	

def camera_orientation(mesh,R_T,i):
	T = np.zeros((4,4))
	T[:3,] = R_T
	T[3,:] = np.array([0,0,0,1])
	new_mesh = copy.deepcopy(mesh).transform(T)
	#print(new_mesh)
	#new_mesh.scale(0.5, center=new_mesh.get_center())
	o3d.io.write_triangle_mesh("mesh"+str(i)+'.ply', new_mesh)
	
	return
	
def PnP(X, p, K, d):
	X = X[:, 0, :]
	p = p.T
	ret, rvecs, t, inliers = cv2.solvePnPRansac(X, p, K, d, cv2.SOLVEPNP_ITERATIVE)
	
	R, _ = cv2.Rodrigues(rvecs)
	
	if inliers is not None:
		p = p[inliers[:,0]]
		X = X[inliers[:,0]]
	
	return R, t, p, X
	
def ReprojectionError(X, pts, Rt, K, homogenity):
	total_error = 0
	R = Rt[:3,:3]
	t = Rt[:3,3]
	
	r, _ = cv2.Rodrigues(R)
	if homogenity == 1:
		X = cv2.convertPointsFromHomogeneous(X.T)
	
	
	p, _ = cv2.projectPoints(X, r, t, K, distCoeffs = None)
	p = p[:, 0, :]
	p = np.float32(p)
	total_error = cv2.norm(p, pts.T, cv2.NORM_L2)
	pts = pts.T
	tot_error = total_error/len(p)
	#print(p, pts.T)
	
	return tot_error, X
	
	
def OptimReprojectionError(X_locs, p, r, t, K):
	total_error = 0
	p = p.T
	num_pts = len(p)
	R = X_locs[0:9].reshape((3,3))
	t = X_locs[9:12]
	K = X_locs[12:21].reshape((3,3))
	X_locs = np.float32(X_locs[21:].reshape((num_pts, 1, 3)))
	error = []
	r, _ = cv2.Rodrigues(R)
	p2d, _ = cv2.projectPoints(X_locs, r, t, K, distCoeffs = None)
	p2d = p2d[:, 0, :]
	#p, _ = cv2.projectPoints(X, r, t, K, distCoeffs = None)
	for idx in range(num_pts):
		img_pt = p[idx]
		reprojected_pt = p2d[idx]
		er = (img_pt - reprojected_pt)**2
		error.append(er)

	
	return np.array(error).ravel()/num_pts
	
def BundleAdjustment(X, p, P, Rt, K):
	num_points = len(p.T)
	R = Rt[:3,:3]
	t = Rt[:3,3]
	opt_variables = np.hstack((R.ravel(), t.ravel()))
	opt_variables = np.hstack((opt_variables,K.ravel()))
	opt_variables = np.hstack((opt_variables, X.ravel()))
	#print(X.shape, p.shape)
	#print(P)
	corrected_values = least_squares(OptimReprojectionError, opt_variables, args = (p, num_points, t, K))
	#P = corrected_values.x[0:12].reshape(3,4)
	#print(P)
	corrected_values = corrected_values.x
	R = corrected_values[0:9].reshape((3,3))
	t = corrected_values[9:12].reshape((3,1))
	K = corrected_values[12:21].reshape((3,3))
	points_3d = corrected_values[21:].reshape((num_points, 1, 3))
	#points_3d = corrected_values.x[12:].reshape((num_points, 3))
	#print(R,t)
	Rt = np.hstack((R,t))
	P = np.matmul(K,Rt)
	
	return Rt, P, points_3d

def to_ply(point_cloud, colors):
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


cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)

K = np.array([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]])
#K = np.array([[1520.400000, 0.000000, 302.320000], [0.000000, 1525.900000, 246.870000], [0.000000, 0.000000, 1.000000]])
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

#img_dir = '/home/arihant/structure-from-motion/'
img_dir = '/home/arihant/structure-from-motion/'

img_list = sorted(os.listdir(img_dir))
images = []
for img in img_list:
	if '.jpg' in img:
		images = images + [img]
i = 0		
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

camera_orientation(mesh,R_t_0,i)

apply_ba = False
while(i < len(images) - 1):
	img0 = cv2.pyrDown(cv2.imread(img_dir + images[i]))
	img1 = cv2.pyrDown(cv2.imread(img_dir + images[i + 1]))
	#img0 = cv2.imread(img_dir + images[i])
	#img1 = cv2.imread(img_dir + images[i + 1])
	
	img0gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
	img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	
	sift = cv2.xfeatures2d.SIFT_create()
	kp0, des0 = sift.detectAndCompute(img0gray, None)
	kp1, des1 = sift.detectAndCompute(img1gray, None)
	
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des0, des1, k = 2)
	
	good = []
	for m,n in matches:
		if m.distance < 0.70 * n.distance:
			good.append(m)
			
	pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
	pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
	
	E, mask = cv2.findEssentialMat(pts0, pts1, K, method = cv2.RANSAC, prob = 0.999, threshold = 0.4, mask = None) 
	
	pts0 = pts0[mask.ravel() == 1]
	pts1 = pts1[mask.ravel() == 1]
	
	_, R, t, mask = cv2.recoverPose(E, pts0, pts1, K)
	
	R_t_1[:3,:3] = np.matmul(R, R_t_0[:3,:3])
	R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3,:3],t.ravel())
	
	camera_orientation(mesh,R_t_1,i+1)
	
	P2 = np.matmul(K, R_t_1)
	
	pts0, pts1, points_3d = Triangulation(P1, P2, pts0, pts1, K, repeat = False)
	
	
	#print(P1, P2)
	
	error, points_3d = ReprojectionError(points_3d, pts1, R_t_1, K, homogenity = 1)
	print("Reprojection Error: ",error)
	
	Rot, trans, pnew, Xnew = PnP(points_3d, pts1, K, np.zeros((5,1), dtype = np.float32))
	pnew = pnew.T
	#print(Rot, trans, pnew.shape, Xnew.shape)
	Rtnew = np.hstack((Rot, trans))
	Pnew = np.matmul(K, Rtnew)
	print(Pnew)
	
	pts0, pts1, points_3d = Triangulation(Pref, Pnew, pts0, pts1, K, repeat = True)
	#print(points_3d[:, 0, :])
	#print(pts1_reg.shape, points_3d.shape)
	
	#x = np.concatenate((x, points_3d[0]))
	#y = np.concatenate((y, points_3d[1]))
	#z = np.concatenate((z, points_3d[2]))
	#print(points_3d.shape)
	error, points_3d = ReprojectionError(points_3d, pts1, R_t_1, K, homogenity = 1)
	
	Xtot = np.vstack((Xtot, points_3d[:, 0, :]))
	pts1_reg = np.array(pts1, dtype = np.int32)
	colors = np.array([img1[l[1],l[0]] for l in pts1_reg.T])
	colorstot = np.vstack((colorstot, colors))
	if apply_ba == True:	
		R_t_1, P2, points_3d = BundleAdjustment(points_3d, pts1, P2, R_t_1, K)
		error, points_3d = ReprojectionError(points_3d, pts1, R_t_1, K, homogenity = 0)
		print("Minimized Reprojection Error: ",error)
	
	cv2.imshow('image1', img0)
	cv2.imshow('image2', img1)
	i = i + 1
	R_t_0 = np.copy(R_t_1)
	P1 = np.copy(P2)
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
		
cv2.destroyAllWindows()
#print(Xtot.shape, colorstot.shape)
print("Processing Point Cloud...")
to_ply(Xtot, colorstot)
print("Done!")
