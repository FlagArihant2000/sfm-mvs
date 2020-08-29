import cv2
import numpy as np
import os


def Triangulation(P1, P2, pts1, pts2, K):
	
	points1 = np.transpose(pts1)
	points2 = np.transpose(pts2)
	
	cloud = cv2.triangulatePoints(P1, P2, points1, points2)
	cloud = cloud / cloud[3]
	
	return points1, points2, cloud
	
def ReprojectionError(X, pts, Rt, K):
	total_error = 0
	R = Rt[:3,:3]
	t = Rt[:3,3]
	
	r, _ = cv2.Rodrigues(R)
	X = cv2.convertPointsFromHomogeneous(X.T)
	
	p, _ = cv2.projectPoints(X, r, t, K, distCoeffs = None)
	
	p = p[:, 0, :]
	#print(pts.shape, p.shape)
	
	error = cv2.norm(p, pts.T, cv2.NORM_L2)/len(p)
	
	tot_error = error ** 2
	tot_error = tot_error / len(X)
	
	return tot_error, X

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

K = np.array([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]])
downscale = 2

K[0,0] = K[0,0] / float(downscale)
K[1,1] = K[1,1] / float(downscale)
K[0,2] = K[0,2] / float(downscale)
K[1,2] = K[1,2] / float(downscale)


R_t_0 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
R_t_1 = np.empty((3,4))

P1 = np.matmul(K, R_t_0)
P2 = np.empty((3,4))


Xtot = np.zeros((1,3))
colorstot = np.zeros((1,3))

img_dir = '/home/arihant/structure-from-motion/'

img_list = sorted(os.listdir(img_dir))
images = []
for img in img_list:
	if '.jpg' in img:
		images = images + [img]
i = 0		
while(i < len(images) - 1):
	img0 = cv2.pyrDown(cv2.imread(img_dir + images[i]))
	img1 = cv2.pyrDown(cv2.imread(img_dir + images[i + 1]))
	
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
	
	P2 = np.matmul(K, R_t_1)
	
	pts0, pts1, points_3d = Triangulation(P1, P2, pts0, pts1, K)
	
	#print(P1, P2)
	error, points_3d = ReprojectionError(points_3d, pts1, R_t_1, K)
	
	Xtot = np.vstack((Xtot, points_3d[:, 0, :]))
	
	pts1_reg = np.array(pts1, dtype = np.int32)
	colors = np.array([img1[l[1],l[0]] for l in pts1_reg.T])
	colorstot = np.vstack((colorstot, colors))
	
	print("Reprojection Error: ",error)
	#print(pts1_reg.shape, points_3d.shape)
	
	#x = np.concatenate((x, points_3d[0]))
	#y = np.concatenate((y, points_3d[1]))
	#z = np.concatenate((z, points_3d[2]))
	
	
	
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
