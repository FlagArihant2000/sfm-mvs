import cv2
import numpy as np
import open3d as o3d

with open('/home/arihant/structure-from-motion/intrinsics.txt') as f:
	lines = f.readlines()
K = np.array([l.strip().split(' ') for l in lines], dtype=np.float32)
downscale = 2
K[0,0] = K[0,0] / float(downscale)
K[1,1] = K[1,1] / float(downscale)
K[0,2] = K[0,2] / float(downscale)
K[1,2] = K[1,2] / float(downscale)

cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)

img1 = cv2.imread('/home/arihant/structure-from-motion/img1.ppm')
img1 = cv2.resize(img1, (0,0), fx = 1/float(downscale), fy = 1/float(downscale), interpolation = cv2.INTER_LINEAR)

img2 = cv2.imread('/home/arihant/structure-from-motion/img2.ppm')
img2 = cv2.resize(img2, (0,0), fx = 1/float(downscale), fy = 1/float(downscale), interpolation = cv2.INTER_LINEAR)

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
matches = flann.knnMatch(des1, des2, k = 2)

good = []
pts1 = []
pts2 = []
for counter, (m,n) in enumerate(matches):
	if m.distance < 0.8 * n.distance:
		good.append(m)
		pts1.append(kp1[m.queryIdx].pt)
		pts2.append(kp2[m.trainIdx].pt)
pts1 = np.array(pts1).reshape(-1,1,2)
pts2 = np.array(pts2).reshape(-1,1,2)


E, mask = cv2.findEssentialMat(pts1, pts2, K, method = cv2.RANSAC, prob = 0.999, threshold = 1.0, mask = None)

pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)


E1 = np.eye(3,4)
E2 = np.hstack((R, t))
P1 = np.matmul(K, E1)
P2 = np.matmul(K, E2)

pts1 = cv2.undistortPoints(pts1, cameraMatrix = K, distCoeffs = None).reshape(-1, 2)
pts2 = cv2.undistortPoints(pts2, cameraMatrix = K, distCoeffs = None).reshape(-1, 2)
cloud4d = cv2.triangulatePoints(E1, E2, pts1.T, pts2.T)
cloud = cloud4d / np.tile(cloud4d[-1,:],(4,1))
cloud = cloud[:3,:].T



cv2.imshow('image1', img1)
cv2.imshow('image2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(cloud)
o3d.io.write_point_cloud('pc.ply',pcd)
print('DONE! Open pc.ply with meshlab for point cloud.')
