# MULTIVIEW STEREO
# AUTHORS: ARIHANT GAUR AND SAURABH KEMEKAR
# ORGANIZATION: IvLabs, VNIT
# Current Under Progress
# TODO Obtain non - occluding points and add patches.

import cv2
import numpy as np
import open3d as o3d
import os

def stringextract(m, i = 1):
	m = m.split(' ')
	m = m[2:]
	arr = []
	if '' in m:
		m.remove('')
	if '[' in m:
		m.remove('[')
	for count, element in enumerate(m):
		if element != '':
			if count == 0:
				arr = arr + [float(element[1:])]
			elif count == len(m) - 1:
				arr = arr + [float(element[:-2])]
			else:
				arr = arr + [float(element)]
	
	arr = np.array(arr)
	if i == 1:
		arr = arr.reshape((3,3))
	else:
		arr = arr.reshape((3,4))
	return arr
	
			



cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)


f = open('poses.txt','r')

K = f.readline()
K = stringextract(K)

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


img_dir = '/home/arihant/sfm-mvs/'

img_list = sorted(os.listdir(img_dir))
images = []
for img in img_list:
	if '.jpg' in img:
		images = images + [img]
i = 0		
#mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

#camera_orientation(mesh,R_t_0,i)
pcd_load = o3d.io.read_point_cloud('sparse.ply')
xyz_load = np.asarray(pcd_load.points)
xyz_color = np.asarray(pcd_load.colors)

xyz_h = cv2.convertPointsToHomogeneous(xyz_load)
xyz_h = xyz_h[:, 0, :]

Rt0 = f.readline()
Rt0 = stringextract(Rt0, 0)
R = Rt0[:3, :3]
t = Rt0[:,3]
t = np.array([[t[0]], [t[1]], [t[2]]])
C0 = -np.matmul(np.linalg.inv(R),t) # O(I)

# img0 will be considered as the reference image and will be varied after every iteration

while(i < len(images) - 1):
	img0 = cv2.pyrDown(cv2.imread(img_dir + images[i])) # R(p)
	img1 = cv2.pyrDown(cv2.imread(img_dir + images[i + 1]))
	
	Rt1 = f.readline()
	Rt1 = stringextract(Rt1, 0)
	R = Rt1[:3, :3]
	t = Rt1[:,3]
	t = np.array([[t[0]], [t[1]], [t[2]]])
	C1 = -np.matmul(np.linalg.inv(R),t) # OC
	
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
	
	P0 = np.matmul(K, Rt0)
	P1 = np.matmul(K, Rt1)
	

	#cloud = cv2.triangulatePoints(P0, P1, pts0.T, pts1.T)
	#cloud = cloud / cloud[3]
	#cloud = cv2.convertPointsFromHomogeneous(cloud.T)
	#cloud = cloud[:, 0, :] # Each point will be considered as c(p)
	#print(cloud.shape)
	#print(C0, cloud.shape)
	
	# Creating normal vectors from center of camera
	#nor = cloud - C0.T
	#row_sums = nor.sum(axis = 1)
	#nor = nor / row_sums[:, np.newaxis]
	#nor = np.nan_to_num(nor) # n(p)
	#print(nor)
	

	cv2.imshow('image1', img0)
	cv2.imshow('image2', img1)
	i = i + 1
	Rt0 = Rt1
	C0 = C1
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
	

cv2.destroyAllWindows()



