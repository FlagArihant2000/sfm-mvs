import cv2
import numpy as np
import os

def RtToR_t(matrix, initial):
	#print(len(matrix))
	print(matrix)
	if initial == 0:
		R = np.array([[float(matrix[2][1:]), float(matrix[3]), float(matrix[4])], [float(matrix[6]), float(matrix[7]), float(matrix[8])], [float(matrix[10]), float(matrix[11]), float(matrix[12])]], dtype = np.float32)
		t = np.array([[float(matrix[5])], [float(matrix[9])], [float(matrix[13][0:-2])]], dtype = np.float32)
	else:
		R = np.array([[float(matrix[3]), float(matrix[5]), float(matrix[6])], [float(matrix[9]), float(matrix[12]), float(matrix[13])], [float(matrix[16]), float(matrix[18]), float(matrix[20])]], dtype = np.float32)
		t = np.array([[float(matrix[8])], [float(matrix[14])], [float(matrix[22][0:-2])]], dtype = np.float32)

	return R, t



cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)

#K = np.array([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]])
#K = np.array([[1520.400000, 0.000000, 302.320000], [0.000000, 1525.900000, 246.870000], [0.000000, 0.000000, 1.000000]])
f = open('poses.txt','r')
K = f.readline() + f.readline()

matrix = K.split(' ')
print(float(matrix[2][1:]))
K = np.array([[float(matrix[2][1:]), float(matrix[3]), float(matrix[4])], [float(matrix[5]), float(matrix[6]), float(matrix[7])], [float(matrix[8]), float(matrix[9]), float(matrix[10][0:-2])]])

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

#posefile = open('poses.txt','w')
#posefile.write("K = " + str(K.ravel()))
#posefile.write("\n")
#posefile.write(str(i) + " = " + str(R_t_0.ravel()))
#posefile.write("\n")

while(i < len(images) - 1):
	img0 = cv2.pyrDown(cv2.imread(img_dir + images[i]))
	img1 = cv2.pyrDown(cv2.imread(img_dir + images[i + 1]))
	if i == 0:
		Rt = f.readline()
		Rt = Rt.split(' ')
		R, t = RtToR_t(Rt, 0)
	else:
		Rt = f.readline() + f.readline()
		Rt = Rt.split(' ')
		Rt.remove('')
		Rt.remove('[')
		R, t = RtToR_t(Rt, 1)
		
	
	#print(R,t)

	cv2.imshow('image1', img0)
	cv2.imshow('image2', img1)
	i = i + 1
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
	

cv2.destroyAllWindows()



