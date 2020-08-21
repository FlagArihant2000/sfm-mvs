import cv2
import numpy as np
import os
from scipy.optimize import least_squares

def Triangulation(R1, t1, R2, t2, kp0, kp1, K):
	#P0 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
	P0 = np.hstack((R1, t1))
	P0 = K.dot(P0)
	
	P1 = np.hstack((R2, t2))
	P1 = K.dot(P1)
	
	points1 = kp0.reshape(2, -1)
	points2 = kp1.reshape(2, -1)
	cloud = cv2.triangulatePoints(P0, P1, points1, points2).reshape(-1, 4)[:, :3]
	return cloud
	



def ReprojectionError(X, R, t, K, d, p):
	reprojected_point = K.dot(R.dot(X.T) + t)
	reprojected_point = cv2.convertPointsFromHomogeneous(reprojected_point.T)[:, 0, :].T

	p = p[:, 0, :]

	error = np.linalg.norm(p - reprojected_point.T)
	return error
	
def bundle_adjustment(X, p, img, P):
	opt_variables = np.hstack((P.ravel(), X.ravel(order="F")))
	num_points = len(p[0])
	
	corrected_values = least_squares(reprojection_loss, opt_variables, args = (p, num_points))
	
	X = corrected_values.x[12:].reshape((num_points, 4))
	
	return X
	
def PnP(X, p2, K, dist_coeff):
	#Xx = X[0]
	#Xy = X[1]
	#Xz = X[2]
	#Xp = np.array([(Xx[i],Xy[i],Xz[i]) for i in range(len(Xx))],dtype = np.float32)
	#p2x = p2[0]
	#p2y = p2[1]
	#p2 = np.array([(p2x[i],p2y[i]) for i in range(len(p2x))],dtype = np.float32)
	
	p2 = p2[:, 0, :]
	#print(X.shape, p2.shape, dist_coeff.shape)
	
	ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(X, p2, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)
	
	R, J = cv2.Rodrigues(rvecs)
	#print(inliers)
	#p2 = p2[inliers[:,0]]
	#Xp = Xp[inliers[:,0]]
	
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

cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)

files = sorted(os.listdir(os.getcwd()))
images = []
for im in files:
	if '.jpg' in im:
		images = images + [im]

i = 0

K = np.array([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]])
D = np.zeros((5,1), dtype = np.float32)

downscale = 2

K[0,0] = K[0,0] / float(downscale)
K[1,1] = K[1,1] / float(downscale)
K[0,2] = K[0,2] / float(downscale)
K[1,2] = K[1,2] / float(downscale)

sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()

Rtot = np.eye(3)
ttot = np.zeros((3,1), dtype = np.float32)
Rprev = np.eye(3)
tprev = np.zeros((3,1), dtype = np.float32)
Xtot = np.zeros((1,3))
colorstot = np.zeros((1,3))
while(i < len(images) - 1):
	img1 = cv2.pyrDown(cv2.imread(images[i]))
	img2 = cv2.pyrDown(cv2.imread(images[i + 1]))
	img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	
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
	pts1 = pts1[mask.ravel() == 1]
	pts2 = pts2[mask.ravel() == 1]
	_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
	#print(R, t) 
	
	#X = TriangulateTwoViews(R, t, pts1, pts2, K)
	if i == 0:
		Rtot = R
		ttot = t
	
	X = Triangulation(Rprev, tprev, Rtot, ttot, pts1, pts2, K)
	Xtot = np.vstack((Xtot, X))
	#print(R, t)
	#reproj_error = ReprojectionError(X, R, t, K, D, pts2)
	#print(reproj_error)
	Rprev = Rtot
	tprev = ttot
	
	R_next, t_next = PnP(X, pts2, K, D)
	#print(R_next, t_next)
	Rtot = np.matmul(Rtot, R_next)
	ttot = ttot + np.matmul(Rtot,t_next)
	pts2_reg = np.array(pts2[:, 0, :], dtype = np.int32)
	
	colors = np.array([img2[l[1],l[0]] for l in pts2_reg])
	colorstot = np.vstack((colorstot, colors))
	#cv2.imshow('image1', img1)
	#cv2.imshow('image2', img2)
	i = i + 1
	print("ITERATION: ", i)
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
		

cv2.destroyAllWindows()
print("PROCESSING POINT CLOUD...")
to_ply(Xtot, colorstot)
print("DONE!")
	
