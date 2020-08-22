import cv2
import numpy as np
import os

def Triangulation(R1, t1, R2, t2, kp0, kp1, K):
	P0 = np.hstack((R1, t1))
	P1 = np.hstack((R2, t2))
	
	P0 = np.float32(K.dot(P0))
	P1 = np.float32(K.dot(P1))
	#points1 = kp0
	#points2 = kp1
	points1 = kp0.reshape(2, -1)
	points2 = kp1.reshape(2, -1)
	cloud = cv2.triangulatePoints(P0, P1, points1, points2)
	cloud = cloud / cloud[3]
	cloud = cv2.convertPointsFromHomogeneous(cloud.T)
	return cloud

def ReprojectionError(X, R, t, K, p):
	total_error = 0
	r, _ = cv2.Rodrigues(R)
	
	p1, _ = cv2.projectPoints(X, r, t, K, distCoeffs = None)
	
	p1 = p1[:, 0, :]
	#print(p1[0], p[0])
	for i in range(len(p)):
		error = np.sqrt((p[i][0] - p1[i][0])**2 + (p[i][0] - p1[i][0])**2) / len(p1)
		total_error += error
		
	#p1 = p1[:, 0, :]
	#p = p[:, 0, :]
	#reprojection_error = np.linalg.norm(p - p1[0, :])
	error = cv2.norm(p, p1, cv2.NORM_L2)/len(p)
	return error
	


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

#img_directory = '/home/arihant/stereo/MVS/templeSparseRing/'

files = sorted(os.listdir(os.getcwd()))
#files = sorted(os.listdir(img_directory))
images = []
for im in files:
	if '.jpg' in im or '.png' in im:
		images = images + [im]

i = 0
#print(images)

K = np.array([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]])
#K = np.array([[1520.400000, 0.000000, 302.320000], [0.000000, 1525.900000, 246.870000], [0.000000, 0.000000, 1.000000]])
D = np.zeros((5,1), dtype = np.float32)

downscale = 2

K[0,0] = K[0,0] / float(downscale)
K[1,1] = K[1,1] / float(downscale)
K[0,2] = K[0,2] / float(downscale)
K[1,2] = K[1,2] / float(downscale)

Xtot = np.zeros((1,3))
colorstot = np.zeros((1,3))

sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()

img1 = cv2.pyrDown(cv2.imread(images[0]))
img2 = cv2.pyrDown(cv2.imread(images[1]))
#img1 = cv2.imread(images[0])
#img2 = cv2.imread(images[1])

img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

kp1, des1 = sift.detectAndCompute(img1gray, None)
kp2, des2 = sift.detectAndCompute(img2gray, None)

matches = bf.knnMatch(des1, des2, k = 2)
good = []
for m,n in matches:
	if m.distance < 0.70 * n.distance:
		good.append(m)
		
pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
E, mask = cv2.findEssentialMat(pts1, pts2, K, method = cv2.RANSAC, prob = 0.999, threshold = 0.4, mask = None)
#print(E)
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

#print(R, t)

X = Triangulation(np.eye(3), np.zeros((3,1)), R, t, pts1, pts2, K)
#print(X)

#error = ReprojectionError(X, R, t, K, pts1)
#print(X.shape, Xtot.shape)
Xtot = np.vstack((Xtot, X[:, 0, :]))
#print(R, t)
reproj_error = ReprojectionError(X, R, t, K, pts2)
print('Reprojection Error: ',reproj_error)
#print(X.shape)

#R_next, t_next = PnP(X, pts2, K, D)
#print(R_next, t_next)
#Rtot = np.matmul(Rtot, R_next)
#ttot = ttot + np.matmul(Rtot,t_next)
pts2_reg = np.array(pts2, dtype = np.int32)

colors = np.array([img2[l[1],l[0]] for l in pts2_reg])
colorstot = np.vstack((colorstot, colors))

#print(error)

to_ply(Xtot, colorstot)
cv2.imshow('image1', img1)
cv2.imshow('image2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

