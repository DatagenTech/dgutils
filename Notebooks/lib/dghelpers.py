from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from os.path import join
import json
import cv2

# Show an image
def imshow(img):
	plt.figure(figsize=(10, 10))
	plt.imshow(img)
	plt.xticks([]); plt.yticks([])

# Show a set of images in a grid
def plot_grid(fig_shape, input_list, plot_func):
    fig_size = 5 * np.array(fig_shape)
    fig = plt.figure(figsize=fig_size)
    grid = ImageGrid(fig, 111, nrows_ncols=fig_shape, axes_pad=0.)

    for i, ax in enumerate(grid):
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        plot_func(ax, input_list[i])

# Convert points from world coordinates (3D) to image coordinates (2D)
def world_to_img(pts_3d, intrinsic_matrix, extrinsic_matrix):
	assert intrinsic_matrix.shape == (3, 3)
	if len(pts_3d.shape) > 2 or pts_3d.shape[-1] not in (3, 4):
		raise ValueError("Invalid input format. The points should be given either as a 1D or 2D array and should have either 3 or 4 (homogenous) coordinates")
	# If there is only a single point, transform it to a 2D array
	if len(pts_3d.shape) == 1:
		pts_3d = pts_3d[np.newaxis]

	pts_2d_homogeneous = world_to_cam(pts_3d, extrinsic_matrix) @ intrinsic_matrix.T
	pts_2d = pts_2d_homogeneous[..., :2] / pts_2d_homogeneous[..., 2][..., np.newaxis]

	return np.squeeze(pts_2d)

def world_to_cam(pts_3d, extrinsic_matrix):
	assert extrinsic_matrix.shape == (3, 4)
	if len(pts_3d.shape) > 2 or pts_3d.shape[-1] not in (3, 4):
		raise ValueError("Invalid input format. The points should be given either as a 1D or 2D array and should have either 3 or 4 (homogenous) coordinates")
	# If there is only a single point, transform it to a 2D array
	if len(pts_3d.shape) == 1:
		pts_3d = pts_3d[np.newaxis]
	# If not in homogeneous coordinates, transform it to homogeneous
	if pts_3d.shape[1] == 3:
		pts_3d = np.hstack([pts_3d, np.ones((pts_3d.shape[0], 1))])

	extrinsic_matrix = extrinsic_matrix.copy()
	
	pts_camera = pts_3d @ extrinsic_matrix.T
	return np.squeeze(pts_camera)


def show_keypoints(img, keypoints, visible, title, convention='ij'):
	# Convention can be either 'ij' (pixel coordinates) or 'xy' (cartesian coordinates)
	color = np.where(visible[:, np.newaxis], [[0, 1, 0]], [[1, 0, 0]])
	imshow(img)
	if convention == 'ij':
		plt.scatter(keypoints[:, 1], keypoints[:, 0], s=5, c=color)
	elif convention == 'xy':
		plt.scatter(keypoints[:, 0], keypoints[:, 1], s=5, c=color)

	pop_invisible = Patch(color=[1, 0, 0], label='Invisible')
	pop_visible = Patch(color=[0, 1, 0], label='Visible')

	plt.legend(handles=[pop_visible, pop_invisible])
	plt.title(title)


############################################## HIC functions ##############################################
class Node:
    def __init__(self, data, children=None):
        if hasattr(data, 'coords_2d'):
            data = data.coords_2d
        self.data = data
        if children is not None:
            self.children = children
        else:
            self.children = []

def get_kinematic_tree(dp):
	print(dp)
	hands = dp.keypoints.hand
	body = dp.keypoints.body

	for side in ('left', 'right'):
	    get_finger_tree = lambda finger : Node(finger.mcp, [Node(finger.pip, [Node(finger.dip, [Node(finger.tip)])])])
	    
	    # We begin with building the kinematic tree of each of the fingers separately
	    fingers = getattr(hands, side).finger
	    thumb = fingers.thumb
	    thumb_tree = Node(thumb.cmc, [Node(thumb.mcp, [Node(thumb.ip, [Node(thumb.tip)])])])
	    index_tree = get_finger_tree(fingers.index)
	    middle_tree = get_finger_tree(fingers.middle)
	    ring_tree = get_finger_tree(fingers.ring)
	    pinky_tree = get_finger_tree(fingers.pinky)

	    # We can build the hand tree from all of the fingers
	    hand_tree = Node(getattr(hands, side).wrist, [thumb_tree, index_tree, middle_tree, ring_tree, pinky_tree])

	    # We do the same with arms, legs and eyes
	    arm_tree = Node(getattr(body.shoulder, side), [Node(getattr(body.elbow, side), [Node(getattr(body.wrist, side))])])
	    leg_tree = Node(getattr(body.hip, side), [Node(getattr(body.knee, side), [Node(getattr(body.ankle, side), [Node(getattr(body.foot, side).index)])])])

	    eye = getattr(body.eye, side)
	    eye_tree = Node(eye.center, [Node(eye.outer), Node(eye.inner), Node(getattr(body.ear, side))])

	    if side == 'left':
	        left_hand_tree = hand_tree
	        left_arm_tree = arm_tree
	        left_leg_tree = leg_tree
	        left_eye_tree = eye_tree
	    elif side == 'right':
	        right_hand_tree = hand_tree
	        right_arm_tree = arm_tree
	        right_leg_tree = leg_tree
	        right_eye_tree = eye_tree

	# We create two new nodes at the center of the shoulders and the hips
	hips_center_position = np.mean([left_leg_tree.data, right_leg_tree.data], axis=0).astype('int')
	shoulders_center_position = np.mean([left_arm_tree.data, right_arm_tree.data], axis=0).astype('int')

	# We can finally create the whole body and face trees
	lower_body_tree = Node(hips_center_position, [left_leg_tree, right_leg_tree])
	mouth_tree = Node(body.mouth.left, [Node(body.mouth.right)])
	upper_face_tree = Node(body.nose, [left_eye_tree, right_eye_tree])
	body_tree = Node(shoulders_center_position, [lower_body_tree, left_arm_tree, right_arm_tree])
	return body_tree, mouth_tree, upper_face_tree, left_hand_tree, right_hand_tree


nrof_colors = 30
cmap = plt.cm.get_cmap('nipy_spectral', nrof_colors)

def draw_skeleton(img, node, thickness, cmap_idx=0):
	node.data[0] = round(node.data[0])
	node.data[1] = round(node.data[1])
	node.data = node.data.astype(int)
	for child in node.children:
		child.data[0] = round(child.data[0])
		child.data[1] = round(child.data[1])
		child.data = child.data.astype(int)
		color = (np.array(cmap(cmap_idx % nrof_colors))[:3]*255).astype('int').tolist()
		cmap_idx += 1
		cv2.line(img, np.flip(node.data), np.flip(child.data), color=color, thickness=thickness, lineType=cv2.LINE_AA)
		draw_skeleton(img, child, thickness, cmap_idx+1)

def draw_keypoints(img, node, thickness, cmap_idx=0):
	node.data[0] = round(node.data[0])
	node.data[1] = round(node.data[1])
	node.data = node.data.astype(int)
	color = (np.array(cmap(cmap_idx % nrof_colors))[:3] * 255).astype('int').tolist()
	cv2.circle(img, np.flip(node.data), thickness, color, -1)
	for child in node.children:
		child.data[0] = round(child.data[0])
		child.data[1] = round(child.data[1])
		child.data = child.data.astype(int)
		cmap_idx += 1
		draw_keypoints(img, child, thickness, cmap_idx)


def hic_visualize_pose(dp, skeleton=True):
	body, mouth, upper_face, left_hand, right_hand = get_kinematic_tree(dp)

	img = dp.visible_spectrum.copy()

	if skeleton:
		drawing_func = draw_skeleton
	else:
		drawing_func = draw_keypoints

	drawing_func(img, body, 5, cmap_idx=0)
	drawing_func(img, upper_face, 2, cmap_idx=nrof_colors // 5)
	drawing_func(img, mouth, 3, cmap_idx=2 * nrof_colors // 5)
	drawing_func(img, left_hand, 2, cmap_idx=3 * nrof_colors // 5)
	drawing_func(img, right_hand, 2, cmap_idx=4 * nrof_colors // 5)

	return img


def set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def set_axes_equal(ax: plt.Axes):
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


def all_body_keypoints_2d(data_point):
	all_keypoints = np.vstack((data_point.keypoints.face.standard.coords_2d,
                          data_point.keypoints.body.shoulder.left.coords_2d,
                          data_point.keypoints.body.shoulder.right.coords_2d,
                          data_point.keypoints.body.elbow.left.coords_2d,
                          data_point.keypoints.body.elbow.right.coords_2d,
                          data_point.keypoints.body.wrist.left.coords_2d,
                          data_point.keypoints.body.wrist.right.coords_2d,
                          data_point.keypoints.hand.left.finger.index.tip.coords_2d,
                          data_point.keypoints.hand.right.finger.index.tip.coords_2d,
                          data_point.keypoints.body.hip.left.coords_2d,
                          data_point.keypoints.body.hip.right.coords_2d,
                          data_point.keypoints.feet.right.heel.coords_2d,
                          data_point.keypoints.feet.left.heel.coords_2d,
                          data_point.keypoints.feet.right.pinky.coords_2d,
                          data_point.keypoints.feet.left.pinky.coords_2d,
                          ))
	return all_keypoints


def all_body_keypoints_3d(data_point):
	all_keypoints = np.vstack((data_point.keypoints.face.standard.coords_3d,
                          data_point.keypoints.body.shoulder.left.coords_3d,
                          data_point.keypoints.body.shoulder.right.coords_3d,
                          data_point.keypoints.body.elbow.left.coords_3d,
                          data_point.keypoints.body.elbow.right.coords_3d,
                          data_point.keypoints.body.wrist.left.coords_3d,
                          data_point.keypoints.body.wrist.right.coords_3d,
                          data_point.keypoints.hand.left.finger.index.tip.coords_3d,
                          data_point.keypoints.hand.right.finger.index.tip.coords_3d,
                          data_point.keypoints.body.hip.left.coords_3d,
                          data_point.keypoints.body.hip.right.coords_3d,
                          data_point.keypoints.feet.right.heel.coords_3d,
                          data_point.keypoints.feet.left.heel.coords_3d,
                          data_point.keypoints.feet.right.pinky.coords_3d,
                          data_point.keypoints.feet.left.pinky.coords_3d,
                          ))
	return all_keypoints