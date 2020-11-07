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
import matplotlib.pyplot as plt

# Input Camera Intrinsic Parameters
K = np.array([[2393.952166119461, -3.410605131648481e-13, 932.3821770809047], [0, 2398.118540286656, 628.2649953288065], [0, 0, 1]])

# Suppose if computationally heavy, then the images can be downsampled once. Note that downsampling is done in powers of two, that is, 1,2,4,8,...
downscale = 2
K[0,0] = K[0,0] / float(downscale)
K[1,1] = K[1,1] / float(downscale)
K[0,2] = K[0,2] / float(downscale)
K[1,2] = K[1,2] / float(downscale)

# Current Path Directory
path = os.getcwd()

# Input the directory where the images are kept. Note that the images have to be named in order for this particular implementation
#img_dir = path + '/Sample Dataset/'
img_dir = '/home/arihant/Desktop/gustav/'

# A provision for bundle adjustment is added, for the newly added points from PnP, before being saved into point cloud. Note that it is still extremely slow
bundle_adjustment = False

# A function, to downscale the image in case SfM pipeline takes time to execute.
def img_downscale(img, downscale):
	downscale = int(downscale/2)
	i = 1
	while(i <= downscale):
		img = cv2.pyrDown(img)
		i = i + 1
	return img
	
# A function, for triangulation, given the image pair and their corresponding projection matrices
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


# Pipeline for Perspective - n - Point
def PnP(X, p, K, d, p_0, initial):
    # print(X.shape, p.shape, p_0.shape)
    if initial == 1:
        X = X[:, 0, :]
        p = p.T
        p_0 = p_0.T

    ret, rvecs, t, inliers = cv2.solvePnPRansac(X, p, K, d, cv2.SOLVEPNP_ITERATIVE)
    # print(X.shape, p.shape, t, rvecs)
    R, _ = cv2.Rodrigues(rvecs)

    if inliers is not None:
        p = p[inliers[:, 0]]
        X = X[inliers[:, 0]]
        p_0 = p_0[inliers[:, 0]]

    return R, t, p, X, p_0

# Calculation for Reprojection error in main pipeline
def ReprojectionError(X, pts, Rt, K, homogenity):
    total_error = 0
    R = Rt[:3, :3]
    t = Rt[:3, 3]

    r, _ = cv2.Rodrigues(R)
    if homogenity == 1:
        X = cv2.convertPointsFromHomogeneous(X.T)

    p, _ = cv2.projectPoints(X, r, t, K, distCoeffs=None)
    p = p[:, 0, :]
    p = np.float32(p)
    pts = np.float32(pts)
    if homogenity == 1:
        total_error = cv2.norm(p, pts.T, cv2.NORM_L2)
    else:
        total_error = cv2.norm(p, pts, cv2.NORM_L2)
    pts = pts.T
    tot_error = total_error / len(p)
    #print(p[0], pts[0])

    return tot_error, X, p


# Calculation of reprojection error for bundle adjustment
def OptimReprojectionError(x):
	Rt = x[0:12].reshape((3,4))
	K = x[12:21].reshape((3,3))
	rest = len(x[21:])
	rest = int(rest * 0.4)
	p = x[21:21 + rest].reshape((2, int(rest/2)))
	X = x[21 + rest:].reshape((int(len(x[21 + rest:])/3), 3))
	R = Rt[:3, :3]
	t = Rt[:3, 3]
	
	total_error = 0
	
	p = p.T
	num_pts = len(p)
	error = []
	r, _ = cv2.Rodrigues(R)
	
	p2d, _ = cv2.projectPoints(X, r, t, K, distCoeffs = None)
	p2d = p2d[:, 0, :]
	#print(p2d[0], p[0])
	for idx in range(num_pts):
		img_pt = p[idx]
		reprojected_pt = p2d[idx]
		er = (img_pt - reprojected_pt)**2
		error.append(er)
	
	err_arr = np.array(error).ravel()/num_pts
	
	print(np.sum(err_arr))
	#err_arr = np.sum(err_arr)
	

	return err_arr

def BundleAdjustment(points_3d, temp2, Rtnew, K, r_error):

	# Set the Optimization variables to be optimized
	opt_variables = np.hstack((Rtnew.ravel(), K.ravel()))
	opt_variables = np.hstack((opt_variables, temp2.ravel()))
	opt_variables = np.hstack((opt_variables, points_3d.ravel()))

	error = np.sum(OptimReprojectionError(opt_variables))
	corrected_values = least_squares(fun = OptimReprojectionError, x0 = opt_variables, gtol = r_error)

	corrected_values = corrected_values.x
	Rt = corrected_values[0:12].reshape((3,4))
	K = corrected_values[12:21].reshape((3,3))
	rest = len(corrected_values[21:])
	rest = int(rest * 0.4)
	p = corrected_values[21:21 + rest].reshape((2, int(rest/2)))
	X = corrected_values[21 + rest:].reshape((int(len(corrected_values[21 + rest:])/3), 3))
	p = p.T
	
	return X, p, Rt
	

def Draw_points(image, pts, repro):
    if repro == False:
        image = cv2.drawKeypoints(image, pts, image, color=(0, 255, 0), flags=0)
    else:
        for p in pts:
            image = cv2.circle(image, tuple(p), 2, (0, 0, 255), -1)
    return image


def to_ply(path, point_cloud, colors, densify):
    out_points = point_cloud.reshape(-1, 3) * 200
    out_colors = colors.reshape(-1, 3)
    print(out_colors.shape, out_points.shape)
    verts = np.hstack([out_points, out_colors])

    # cleaning point cloud
    mean = np.mean(verts[:, :3], axis=0)
    temp = verts[:, :3] - mean
    dist = np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
    #print(dist.shape, np.mean(dist))
    indx = np.where(dist < np.mean(dist) + 300)
    verts = verts[indx]
    #print( verts.shape)
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
        with open(path + '/Point_Cloud/sparse.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')
    else:
        with open(path + '/Point_Cloud/dense.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')
            
# Camera pose registration. This is quite useful in visualizing the pose for each camera, along with the point cloud. It is currently disabled. Will be fixed later.
def camera_orientation(path, mesh, R_T, i):
    T = np.zeros((4, 4))
    T[:3, ] = R_T
    T[3, :] = np.array([0, 0, 0, 1])
    new_mesh = copy.deepcopy(mesh).transform(T)
    # print(new_mesh)
    #new_mesh.scale(0.5, center=new_mesh.get_center())
    o3d.io.write_triangle_mesh(path + "/Point_Cloud/camerapose" + str(i) + '.ply', new_mesh)
    return


def common_points(pts1, pts2, pts3):
    '''Here pts1 represent the points image 2 find during 1-2 matching
    and pts2 is the points in image 2 find during matching of 2-3 '''
    indx1 = []
    indx2 = []
    for i in range(pts1.shape[0]):
        a = np.where(pts2 == pts1[i, :])
        if a[0].size == 0:
            pass
        else:
            indx1.append(i)
            indx2.append(a[0][0])

    '''temp_array1 and temp_array2 will which are not common '''
    temp_array1 = np.ma.array(pts2, mask=False)
    temp_array1.mask[indx2] = True
    temp_array1 = temp_array1.compressed()
    temp_array1 = temp_array1.reshape(int(temp_array1.shape[0] / 2), 2)

    temp_array2 = np.ma.array(pts3, mask=False)
    temp_array2.mask[indx2] = True
    temp_array2 = temp_array2.compressed()
    temp_array2 = temp_array2.reshape(int(temp_array2.shape[0] / 2), 2)
    print("Shape New Array", temp_array1.shape, temp_array2.shape)
    return np.array(indx1), np.array(indx2), temp_array1, temp_array2

# Feature detection for two images, followed by feature matching
def find_features(img0, img1):
    img0gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp0, des0 = sift.detectAndCompute(img0gray, None)
    
    #lk_params = dict(winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    
    kp1, des1 = sift.detectAndCompute(img1gray, None)
    #pts0 = np.float32([m.pt for m in kp0])
    #pts1, st, err = cv2.calcOpticalFlowPyrLK(img0gray, img1gray, pts0, None, **lk_params)
    #pts0 = pts0[st.ravel() == 1]
    #pts1 = pts1[st.ravel() == 1]
    #print(pts0.shape, pts1.shape)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des0, des1, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            good.append(m)

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])

    return pts0, pts1



cv2.namedWindow('image', cv2.WINDOW_NORMAL)

posearr = K.ravel()
R_t_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
R_t_1 = np.empty((3, 4))

P1 = np.matmul(K, R_t_0)
Pref = P1
P2 = np.empty((3, 4))

Xtot = np.zeros((1, 3))
colorstot = np.zeros((1, 3))


img_list = sorted(os.listdir(img_dir))
images = []
for img in img_list:
    if '.jpg' in img.lower() or '.png' in img.lower():
        images = images + [img]
i = 0
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()



densify = False  # Added in case we will merge densification step in this code. Mostly it will be considered separately, though still added here.

# Setting the Reference two frames
img0 = img_downscale(cv2.imread(img_dir + '/' + images[i]), downscale)
img1 = img_downscale(cv2.imread(img_dir + '/' + images[i + 1]), downscale)

pts0, pts1 = find_features(img0, img1)

# Finding essential matrix
E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)
pts0 = pts0[mask.ravel() == 1]
pts1 = pts1[mask.ravel() == 1]
# The pose obtained is for second image with respect to first image
_, R, t, mask = cv2.recoverPose(E, pts0, pts1, K)  # |finding the pose
pts0 = pts0[mask.ravel() > 0]
pts1 = pts1[mask.ravel() > 0]
R_t_1[:3, :3] = np.matmul(R, R_t_0[:3, :3])
R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3, :3], t.ravel())

P2 = np.matmul(K, R_t_1)

# Triangulation is done for the first image pair. The poses will be set as reference, that will be used for increemental SfM
pts0, pts1, points_3d = Triangulation(P1, P2, pts0, pts1, K, repeat=False)
# Backtracking the 3D points onto the image and calculating the reprojection error. Ideally it should be less than one.
# If found to be the culprit for an incorrect point cloud, enable Bundle Adjustment
error, points_3d, repro_pts = ReprojectionError(points_3d, pts1, R_t_1, K, homogenity = 1)
print("REPROJECTION ERROR: ", error)
Rot, trans, pts1, points_3d, pts0t = PnP(points_3d, pts1, K, np.zeros((5, 1), dtype=np.float32), pts0, initial=1)
#Xtot = np.vstack((Xtot, points_3d))

R = np.eye(3)
t = np.array([[0], [0], [0]], dtype=np.float32)

# Here, the total images to be take into consideration can be varied. Ideally, the whole set can be used, or a part of it. For whole lot: use tot_imgs = len(images) - 2
tot_imgs = len(images) - 2 

posearr = np.hstack((posearr, P1.ravel()))
posearr = np.hstack((posearr, P2.ravel()))

gtol_thresh = 0.5
#camera_orientation(path, mesh, R_t_0, 0)
#camera_orientation(path, mesh, R_t_1, 1)

for i in tqdm(range(tot_imgs)):
    # Acquire new image to be added to the pipeline and acquire matches with image pair
    img2 = img_downscale(cv2.imread(img_dir + '/' + images[i + 2]), downscale)

    # pts0,pts1 = find_features(img1,img2)

    pts_, pts2 = find_features(img1, img2)
    if i != 0:
        pts0, pts1, points_3d = Triangulation(P1, P2, pts0, pts1, K, repeat = False)
        pts1 = pts1.T
        points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
        points_3d = points_3d[:, 0, :]
    
    # There gone be some common point in pts1 and pts_
    # we need to find the indx1 of pts1 match with indx2 in pts_
    indx1, indx2, temp1, temp2 = common_points(pts1, pts_, pts2)
    com_pts2 = pts2[indx2]
    com_pts_ = pts_[indx2]
    com_pts0 = pts0.T[indx1]
    # We have the 3D - 2D Correspondence for new image as well as point cloud obtained from before. The common points can be used to find the world coordinates of the new image
    # using Perspective - n - Point (PnP)
    Rot, trans, com_pts2, points_3d, com_pts_ = PnP(points_3d[indx1], com_pts2, K, np.zeros((5, 1), dtype=np.float32), com_pts_, initial = 0)
    # Find the equivalent projection matrix for new image
    Rtnew = np.hstack((Rot, trans))
    Pnew = np.matmul(K, Rtnew)

    #print(Rtnew)
    error, points_3d, _ = ReprojectionError(points_3d, com_pts2, Rtnew, K, homogenity = 0)
   
    
    temp1, temp2, points_3d = Triangulation(P2, Pnew, temp1, temp2, K, repeat = False)
    error, points_3d, _ = ReprojectionError(points_3d, temp2, Rtnew, K, homogenity = 1)
    print("Reprojection Error: ", error)
    # We are storing the pose for each image. This will be very useful during multiview stereo as this should be known
    posearr = np.hstack((posearr, Pnew.ravel()))

    # If bundle adjustment is considered. gtol_thresh represents the gradient threshold or the min jump in update that can happen. If the jump is smaller, optimization is terminated.
    # Note that most of the time, the pipeline yield a reprojection error less than 0.1! However, it is often too slow, often close to half a minute per frame!
    # For point cloud registration, the points are updated in a NumPy array. To visualize the object, the corresponding BGR color is also updated, which will be merged
    # at the end with the 3D points
    if bundle_adjustment:
        print("Bundle Adjustment...")
        points_3d, temp2, Rtnew = BundleAdjustment(points_3d, temp2, Rtnew, K, gtol_thresh)
        Pnew = np.matmul(K, Rtnew)
        error, points_3d, _ = ReprojectionError(points_3d, temp2, Rtnew, K, homogenity = 0)
        print("Minimized error: ",error)
        Xtot = np.vstack((Xtot, points_3d))
        pts1_reg = np.array(temp2, dtype=np.int32)
        colors = np.array([img2[l[1], l[0]] for l in pts1_reg])
        colorstot = np.vstack((colorstot, colors))
    else:
        Xtot = np.vstack((Xtot, points_3d[:, 0, :]))
        pts1_reg = np.array(temp2, dtype=np.int32)
        colors = np.array([img2[l[1], l[0]] for l in pts1_reg.T])
        colorstot = np.vstack((colorstot, colors)) 
    #camera_orientation(path, mesh, Rtnew, i + 2)    


    R_t_0 = np.copy(R_t_1)
    P1 = np.copy(P2)
    plt.scatter(i, error)
    plt.pause(0.05)

    img0 = np.copy(img1)
    img1 = np.copy(img2)
    pts0 = np.copy(pts_)
    pts1 = np.copy(pts2)
    #P1 = np.copy(P2)
    P2 = np.copy(Pnew)
    cv2.imshow('image', img2)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

plt.show()
cv2.destroyAllWindows()

# Finally, the obtained points cloud is registered and saved using open3d. It is saved in .ply form, which can be viewed using meshlab
print("Processing Point Cloud...")
print(Xtot.shape, colorstot.shape)
to_ply(path, Xtot, colorstot, densify)
print("Done!")
# Saving projection matrices for all the images
np.savetxt('pose.csv', posearr, delimiter = '\n')

