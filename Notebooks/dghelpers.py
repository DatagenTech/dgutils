from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from os.path import join
import json
import cv2

# Show an image
def imshow(img):
	plt.figure(figsize=(10, 10))
	plt.imshow(img)
	plt.xticks([]); plt.yticks([])

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
def filter_2d(d):
	if 'pixel_2d' in d.keys():
		return np.array([d['pixel_2d']['x'], d['pixel_2d']['y']])
	else:
		new_d = {}
		for key, value in d.items():
			new_d[key] = filter_2d(value)
		return new_d

class Node:
	def __init__(self, data, children=None):
		self.data = data
		if children is not None:
			self.children = children
		else:
			self.children = []

def get_kinematic_tree(datapoint_path):
	BODY_KPTS_PATH = join(datapoint_path, 'key_points', 'body_key_points.json')
	HANDS_KPTS_PATH = join(datapoint_path, 'key_points', 'hands_key_points.json')

	with open(BODY_KPTS_PATH) as f:
		body_dict = json.load(f)['body']

	with open(HANDS_KPTS_PATH) as f:
		hands_dict = json.load(f)['hands']

	body_dict = filter_2d(body_dict)
	hands_dict = filter_2d(hands_dict)

	for side in ('left', 'right'):
		get_finger_tree = lambda finger: Node(finger['mcp'],
											  [Node(finger['pip'], [Node(finger['dip'], [Node(finger['tip'])])])])

		# We begin with building the kinematic tree of each of the fingers separately
		fingers_dict = hands_dict[side]['finger']
		thumb_dict = fingers_dict['thumb']
		thumb = Node(thumb_dict['cmc'], [Node(thumb_dict['mcp'], [Node(thumb_dict['ip'], [Node(thumb_dict['tip'])])])])
		index = get_finger_tree(fingers_dict['index'])
		middle = get_finger_tree(fingers_dict['middle'])
		ring = get_finger_tree(fingers_dict['ring'])
		pinky = get_finger_tree(fingers_dict['pinky'])

		# We can build the hand tree from all of the fingers
		hand = Node(hands_dict[side]['wrist'], [thumb, index, middle, ring, pinky])

		# We do the same with arms, legs and eyes
		arm = Node(body_dict['shoulder'][side], [Node(body_dict['elbow'][side], [Node(body_dict['wrist'][side])])])
		leg = Node(body_dict['hip'][side], [
			Node(body_dict['knee'][side], [Node(body_dict['ankle'][side], [Node(body_dict['foot'][side]['index'])])])])

		eye_dict = body_dict['eye'][side]
		eye = Node(eye_dict['center'], [Node(eye_dict['outer']), Node(eye_dict['inner']), Node(body_dict['ear'][side])])

		if side == 'left':
			left_hand = hand
			left_arm = arm
			left_leg = leg
			left_eye = eye
		elif side == 'right':
			right_hand = hand
			right_arm = arm
			right_leg = leg
			right_eye = eye

	# We create two new nodes at the center of the shoulders and the hips
	hips_center_position = np.mean([left_leg.data, right_leg.data], axis=0).astype('int')
	shoulders_center_position = np.mean([left_arm.data, right_arm.data], axis=0).astype('int')

	# We can finally create the whole body and face trees
	lower_body = Node(hips_center_position, [left_leg, right_leg])
	body = Node(shoulders_center_position, [lower_body, left_arm, right_arm])
	mouth = Node(body_dict['mouth']['left'], [Node(body_dict['mouth']['right'])])
	upper_face = Node(body_dict['nose'], [left_eye, right_eye])
	return body, mouth, upper_face, left_hand, right_hand


nrof_colors = 30
cmap = plt.cm.get_cmap('nipy_spectral', nrof_colors)

def draw_skeleton(img, node, thickness, cmap_idx=0):
	for child in node.children:
		color = (np.array(cmap(cmap_idx % nrof_colors))[:3]*255).astype('int').tolist()
		cmap_idx += 1
		cv2.line(img, np.flip(node.data), np.flip(child.data), color=color, thickness=thickness, lineType=cv2.LINE_AA)
		draw_skeleton(img, child, thickness, cmap_idx+1)

def draw_keypoints(img, node, thickness, cmap_idx=0):
	color = (np.array(cmap(cmap_idx % nrof_colors))[:3] * 255).astype('int').tolist()
	cv2.circle(img, np.flip(node.data), thickness, color, -1)
	for child in node.children:
		cmap_idx += 1
		draw_keypoints(img, child, thickness, cmap_idx)


def hic_visualize_pose(datapoint_path, skeleton=True):
	RGB_IMG_PATH = join(datapoint_path, 'visible_spectrum.png')
	body, mouth, upper_face, left_hand, right_hand = get_kinematic_tree(datapoint_path)

	img = cv2.imread(RGB_IMG_PATH)

	if skeleton:
		drawing_func = draw_skeleton
	else:
		drawing_func = draw_keypoints

	drawing_func(img, body, 5, cmap_idx=0)
	drawing_func(img, upper_face, 2, cmap_idx=nrof_colors // 5)
	drawing_func(img, mouth, 3, cmap_idx=2 * nrof_colors // 5)
	drawing_func(img, left_hand, 2, cmap_idx=3 * nrof_colors // 5)
	drawing_func(img, right_hand, 2, cmap_idx=4 * nrof_colors // 5)

	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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