import cv2
import numpy as np
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


img_dir = '/home/arihant/structure-from-motion/'

img_list = sorted(os.listdir(img_dir))
images = []
for img in img_list:
	if '.jpg' in img:
		images = images + [img]
i = 0		
#mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

#camera_orientation(mesh,R_t_0,i)


while(i < len(images) - 1):
	img0 = cv2.pyrDown(cv2.imread(img_dir + images[i]))
	img1 = cv2.pyrDown(cv2.imread(img_dir + images[i + 1]))
	
	Rt = f.readline()
	Rt = stringextract(Rt, i = 0)
	R = Rt[:3, :3]
	t = Rt[:,3]
	t = np.array([[t[0]], [t[1]], [t[2]]])
	
		
	
	#print(R,t)

	cv2.imshow('image1', img0)
	cv2.imshow('image2', img1)
	i = i + 1
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
	

cv2.destroyAllWindows()



