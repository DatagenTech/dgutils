from matplotlib import pyplot as plt
import numpy as np

# Show an image
def imshow(img):
	plt.figure(figsize=(10, 10))
	plt.imshow(img)
	plt.xticks([]); plt.yticks([])

# Convert points from world coordinates (3D) to image coordinates (2D)
def world_to_img(pts_3d, intrinsic_matrix, extrinsic_matrix):
	if len(pts_3d.shape) > 2 or pts_3d.shape[-1] not in (3, 4):
		raise ValueError("Invalid input format. The points should be given either as a 1D or 2D array and should have either 3 or 4 (homogenous) coordinates")
	# If there is only a single point, transform it to a 2D array
	if len(pts_3d.shape) == 1:
		pts_3d = pts_3d[np.newaxis]
	# If not in homogeneous coordinates, transform it to homogeneous
	if pts_3d.shape[1] == 3:
		pts_3d = np.hstack([pts_3d, np.ones((68, 1))])

	extrinsic_matrix = extrinsic_matrix.copy()
	# FIXME: Used to fix the extrinsic matrix bug. Remove that once fixed
	extrinsic_matrix[:, 3] *= -1
	extrinsic_matrix_homogeneous = np.vstack([extrinsic_matrix, [0,0,0,1]])
	pts_camera_homogeneous = pts_3d @ extrinsic_matrix_homogeneous.T
	intrinsic_matrix_homogeneous = np.hstack([intrinsic_matrix, np.zeros((3, 1))])
	pts_2d_homogeneous = pts_camera_homogeneous @ intrinsic_matrix_homogeneous.T
	pts_2d = pts_2d_homogeneous[:, :2] / pts_2d_homogeneous[:, 2][:, np.newaxis]
	return pts_2d