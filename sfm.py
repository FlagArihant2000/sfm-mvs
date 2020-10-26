# Structure from Motion
# Authors: Arihant Gaur and Saurabh Kemekar
# Organization: IvLabs, VNIT


import cv2
import numpy as np
import os
from scipy.optimize import least_squares
import copy
import open3d as o3d
from tqdm import tqdm
#from features import *
import matplotlib.pyplot as plt

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


def camera_orientation(path,mesh,R_T,i):
	T = np.zeros((4,4))
	T[:3,] = R_T
	T[3,:] = np.array([0,0,0,1])
	new_mesh = copy.deepcopy(mesh).transform(T)
	#print(new_mesh)
	#new_mesh.scale(0.5, center=new_mesh.get_center())
	o3d.io.write_triangle_mesh(path+"/Point_Cloud/camerapose"+str(i)+'.ply', new_mesh)
	return

def PnP(X, p, K, d, p_0, initial):
	#print(X.shape, p.shape, p_0.shape)
	if initial == 1:
		X = X[:, 0, :]
		p = p.T
		p_0 = p_0.T
	
	ret, rvecs, t, inliers = cv2.solvePnPRansac(X, p, K, d, cv2.SOLVEPNP_ITERATIVE)
	#print(X.shape, p.shape, t, rvecs)
	R, _ = cv2.Rodrigues(rvecs)

	if inliers is not None:
		p = p[inliers[:,0]]
		X = X[inliers[:,0]]
		p_0 = p_0[inliers[:,0]]

	return R, t, p, X, p_0

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
	pts = np.float32(pts)
	if homogenity == 1:
		total_error = cv2.norm(p, pts.T, cv2.NORM_L2)
	else:
		total_error = cv2.norm(p, pts, cv2.NORM_L2)
	pts = pts.T
	tot_error = total_error/len(p)
	#print(p, pts.T)

	return tot_error, X, p

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

def BundleAdjustment(X,pts1, P, Rt, K):
	print(X.shape,pts1.shape)
	num_points = len(pts1.T)
	R = Rt[:3,:3]
	t = Rt[:3,3]
	opt_variables = np.hstack((R.ravel(), t.ravel()))
	opt_variables = np.hstack((opt_variables,K.ravel()))
	opt_variables = np.hstack((opt_variables, X.ravel()))
	print("The Size of opt_variables=",opt_variables.shape)

	corrected_values = least_squares(OptimReprojectionError, opt_variables, args = (pts1, num_points, t, K))

	corrected_values = corrected_values.x
	R = corrected_values[0:9].reshape((3,3))
	t = corrected_values[9:12].reshape((3,1))
	K = corrected_values[12:21].reshape((3,3))
	points_3d = corrected_values[21:].reshape((num_points, 1, 3))
	Rt = np.hstack((R,t))
	P = np.matmul(K,Rt)

	return Rt, P, points_3d

def Draw_points(image,pts, repro):
	if repro == False:
		image = cv2.drawKeypoints(image,pts,image, color=(0, 255, 0),flags=0)
	else:
		for p in pts:
			image = cv2.circle(image,tuple(p), 2, (0, 0, 255), -1)
	return image


def to_ply(path,point_cloud, colors, densify):
	out_points = point_cloud.reshape(-1,3)*200
	out_colors = colors.reshape(-1,3)
	print(out_colors.shape,out_points.shape)
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

def common_points(pts1,pts2,pts3):

	'''Here pts1 represent the points image 2 find during 1-2 matching
	and pts2 is the points in image 2 find during matching of 2-3 '''
	indx1 = []
	indx2 = []
	for i in range(pts1.shape[0]):
		a = np.where(pts2 == pts1[i,:])
		if a[0].size == 0:
			pass
		else:
			indx1.append(i)
			indx2.append(a[0][0])


	'''temp_array1 and temp_array2 will which are not common '''
	temp_array1 = np.ma.array(pts2,mask = False)
	temp_array1.mask[indx2] = True
	temp_array1 = temp_array1.compressed()
	temp_array1 = temp_array1.reshape(int(temp_array1.shape[0]/2),2)


	temp_array2 = np.ma.array(pts3,mask = False)
	temp_array2.mask[indx2] = True
	temp_array2 = temp_array2.compressed()
	temp_array2 = temp_array2.reshape(int(temp_array2.shape[0] / 2), 2)
	print("Shape New Array",temp_array1.shape,temp_array2.shape)
	return  np.array(indx1), np.array(indx2),temp_array1,temp_array2

def find_features(img0,img1):
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

	return pts0,pts1
# cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
# cv2.namedWindow('image2', cv2.WINDOW_NORMAL)

#K = np.array([[2.39395217e+03, 0.00000000e+00, 9.32382177e+02],
#       [0.00000000e+00, 2.39811854e+03, 6.28264995e+02],
#       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
       
#K = np.array([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]])
K = np.array([[2393.952166119461, -3.410605131648481e-13, 932.3821770809047], [0, 2398.118540286656, 628.2649953288065], [0, 0, 1]])
#K = np.array([[1520.400000, 0.000000, 302.320000], [0.000000, 1525.900000, 246.870000], [0.000000, 0.000000, 1.000000]])
downscale = 2

K[0,0] = K[0,0] / float(downscale)
K[1,1] = K[1,1] / float(downscale)
K[0,2] = K[0,2] / float(downscale)
K[1,2] = K[1,2] / float(downscale)
#K = np.array([[1.19697608e+03, -3.41060513e-13, 4.66191089e+02], [0.00000000e+00, 1.19905927e+03, 3.14132498e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
R_t_0 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
R_t_1 = np.empty((3,4))

P1 = np.matmul(K, R_t_0)
Pref = P1
P2 = np.empty((3,4))



Xtot = np.zeros((1,3))
colorstot = np.zeros((1,3))

path = os.getcwd()
#img_dir = path + '/Sample Dataset/'
img_dir = '/home/arihant/Desktop/Sequential-Structure-from-Motion/images/'
#img_dir = '/home/arihant/Downloads/templeRing/'
#img_dir = '/home/arihant/Desktop/uoft/'



img_list = sorted(os.listdir(img_dir))
images = []
for img in img_list:
	if '.jpg' in img.lower() or '.png' in img.lower():
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

apply_ba =False

densify  = False # Application of Patch based MVS to make a denser point cloud

# Setting the Reference two frames
if downscale == 1:
	img0 = cv2.imread(img_dir + '/' + images[i])
	img1 = cv2.imread(img_dir +'/'+ images[i + 1])
else:
	img0 = cv2.pyrDown(cv2.imread(img_dir +'/'+ images[i]))
	img1 = cv2.pyrDown(cv2.imread(img_dir +'/'+ images[i + 1]))
	
pts0, pts1 = find_features(img0, img1)
E, mask = cv2.findEssentialMat(pts0, pts1, K, method = cv2.RANSAC, prob = 0.999, threshold = 0.4, mask = None)
pts0 = pts0[mask.ravel() == 1]
pts1 = pts1[mask.ravel() == 1]
_, R, t, mask = cv2.recoverPose(E, pts0, pts1, K)# |finding the pose
pts0 = pts0[mask.ravel() > 0]
pts1 = pts1[mask.ravel() > 0]
R_t_1[:3,:3] = np.matmul(R, R_t_0[:3,:3])
R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3,:3],t.ravel())

P2 = np.matmul(K, R_t_1)
#print(P1,P2)
pts0, pts1, points_3d = Triangulation(P1, P2, pts0, pts1, K, repeat = False)
error, points_3d, repro_pts = ReprojectionError(points_3d, pts1, R_t_1, K, homogenity = 1)
print("REPROJECTION ERROR: ", error)
Rot, trans, pts1, points_3d, pts0t = PnP(points_3d, pts1, K, np.zeros((5,1), dtype = np.float32), pts0, initial = 1)
#Rtnew = np.hstack((Rot, trans))
#np.savetxt('p.txt',Rtnew)
#R_t_1[:3,:3] = np.matmul(R, R_t_0[:3,:3])
#R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3,:3],t.ravel())
#np.savetxt('p.txt',R_t_1)
#P2 = np.matmul(K, R_t_1)
R = np.eye(3)
t = np.array([[0],[0],[0]], dtype = np.float32)
zoom = 1
for i in tqdm(range(9)):
	if downscale == 1:
		img2 = cv2.imread(img_dir +'/'+ images[i + 2])
	else:
		img2 = cv2.pyrDown(cv2.imread(img_dir +'/'+ images[i + 2]))
	#image = img1.copy()

	#pts0,pts1 = find_features(img1,img2)
	
	pts_, pts2 = find_features(img1, img2)
	if i != 0:
		#print(pts0.shape, pts1.shape, pts2.shape)
		#E, mask = cv2.findEssentialMat(pts0, pts1, K, method = cv2.RANSAC, prob = 0.999, threshold = 0.4, mask = None)
		#pts0 = pts0[mask.ravel() == 1]
		#pts1 = pts1[mask.ravel() == 1]
		#_, Rtemp, ttemp, mask = cv2.recoverPose(E, pts0, pts1, K)
		#Ptemp = np.hstack((Rtemp,ttemp))
		#Ptemp = np.matmul(K, Ptemp)
		#print(P1, P2)
		pts0, pts1, points_3d = Triangulation(P1, P2, pts0, pts1, K, repeat = False)
		pts1 = pts1.T
		points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
		points_3d = points_3d[:, 0, :]
	# There gone be some common point in pts1 and pts_
	# we need to find the indx1 of pts1 match with indx2 in pts_
	#print(pts1.shape, pts_.shape, pts2.shape)
	indx1,indx2,temp1,temp2 = common_points(pts1, pts_, pts2)
	com_pts2 = pts2[indx2]
	com_pts_ = pts_[indx2]
	
	Rot, trans, com_pts2, points_3d, com_pts_ = PnP(points_3d[indx1], com_pts2, K, np.zeros((5,1), dtype = np.float32), com_pts_, initial = 0)
	if i == 0:
		R = np.matmul(R, Rot)
		t = trans

	else:
		#R = np.matmul(R, Rot)
		R = Rot
		t = trans #+ np.matmul(R, trans)
	
	Rtnew = np.hstack((R,t))
	Pnew = np.matmul(K, Rtnew)
	
	
	np.savetxt('p.txt',Rtnew)
	print(Rtnew)
	error, points_3d, _ = ReprojectionError(points_3d, com_pts2, Rtnew, K, homogenity = 0)
	print("Reprojection Error for image 2 and 3: ", error)
	temp1,temp2, points_3d = Triangulation(P2, Pnew, temp1, temp2, K, repeat = False)
	error, points_3d,_ = ReprojectionError(points_3d, temp2, Rtnew, K, homogenity = 1)

	#posefile.write(str(i + 1) + " = " + str(Rtnew.flatten()).replace('\n',''))
	#posefile.write("\n")
	print("New error",error,points_3d.shape,Xtot.shape)
	Xtot = np.vstack((Xtot, points_3d[:, 0, :] * zoom))
	pts1_reg = np.array(temp2, dtype = np.int32)
	
	if not densify:
		colors = np.array([img2[l[1],l[0]] for l in pts1_reg.T])
	else:
		colors = np.array([img2[l[0],l[1]] for l in pts1_reg])
	colorstot = np.vstack((colorstot, colors))
	print('oeirnhguhebygb6tf', pts1_reg.shape, colors.shape)
	if apply_ba:
		R_t_1, P2, points_3d = BundleAdjustment(points_3d,pts1,P2, R_t_1, K)
		error, points_3d = ReprojectionError(points_3d, pts1, R_t_1, K, homogenity = 0)
		print("Minimized Reprojection Error: ",error)

	R_t_0 = np.copy(R_t_1)
	P1 = np.copy(P2)
	plt.scatter(i, error)
	plt.pause(0.05)
	#cv2.imwrite('image.jpg',image)
	
	img0 = np.copy(img1)
	img1 = np.copy(img2)
	pts0 = np.copy(pts_)
	pts1 = np.copy(pts2)
	P1 = np.copy(P2)
	P2 = np.copy(Pnew)
	
	if cv2.waitKey(1) & 0xff == ord('q'):
		break

plt.show()
cv2.destroyAllWindows()

print("Processing Point Cloud...")
print(Xtot.shape,colorstot.shape)
to_ply(path,Xtot, colorstot, densify)
print("Done!")

posefile.close()
