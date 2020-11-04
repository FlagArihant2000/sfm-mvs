
import cv2
import numpy as np
import os
import copy
import open3d as o3d

def camera_orientation(path, mesh, R_T, i):
    T = np.zeros((4, 4))
    T[:3, ] = R_T
    T[3, :] = np.array([0, 0, 0, 1])
    new_mesh = copy.deepcopy(mesh).transform(T)
    # print(new_mesh)
    new_mesh.scale(0.2, center=new_mesh.get_center())
    o3d.io.write_triangle_mesh(path + "/Point_Cloud/camerapose" + str(i) + '.ply', new_mesh)
    return

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
    verts = np.hstack([out_points, out_colors])

    # cleaning point cloud
    mean = np.mean(verts[:, :3], axis=0)
    temp = verts[:, :3] - mean
    dist = np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
    indx = np.where(dist < np.mean(dist) + 300)
    verts = verts[indx]
    name = 'sparse5.ply'
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
        with open(path + '/Point_Cloud/' + name , 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')
            print("Point cloud Successfully Saved {} ".format(name))
    else:
        with open(path + '/Point_Cloud/dense.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')