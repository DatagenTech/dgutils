from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from os.path import join
import json
import cv2
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev
from matplotlib.path import Path
import datagen



# Show an image
def imshow(img):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.xticks([]); plt.yticks([])


def imshow_in_frame(im, figsize=(9, 9), ticks:bool=False, **kwargs):
    plt.figure(figsize=figsize)
    if ticks is False:
        plt.xticks([])
        plt.yticks([])
    plt.imshow(im, **kwargs)


# Show a set of images in a grid
def plot_grid(fig_shape, input_list, plot_func):
    fig_size = 5 * np.array(fig_shape)
    fig = plt.figure(figsize=fig_size)
    grid = ImageGrid(fig, 111, nrows_ncols=fig_shape, axes_pad=0.)

    for i, ax in enumerate(grid):
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        plot_func(ax, input_list[i])


# This function will let us perform the operation while eliminating the potential artifacts at the borders of the mask
def blend_images(background, foreground, mask, sigma=2):
    if type(mask) == np.ndarray:
        mask = gaussian_filter(mask.astype(float), sigma=sigma)
        if len(mask.shape) == 2:
            mask = mask[..., None]
    return (background * (1 - mask) + foreground * mask).astype(np.uint8)


# This function generates binary masks for pupils, iris, and eyeballs (even with glasses)
def eyes_masks(dp):
    # Creates an eyeball mask even when we have glasses
    def eyeball_mask(dp):
        right_eye_upper_idx = [33, 246, 161, 160, 159, 158, 157, 173, 133]  # Outside-in (including corners)
        right_eye_lower_idx = [155, 154, 153, 145, 144, 163, 7]  # Inside-out (excluding corners)
        left_eye_upper_idx = [362, 398, 384, 385, 386, 387, 388, 466, 263]  # Outside-in (including corners)
        left_eye_lower_idx = [390, 373, 374, 380, 381, 382]  # Inside-out (excluding corners)

        def mask_from_kpts(kpts):
            def continuous_kpts(control_pts):
                # This function interpolates the given keypoints to smoothly populate the spaces between them
                _, idx = np.unique(control_pts, axis=0, return_index=True)
                control_pts = control_pts[np.sort(idx)]
                tck, u = splprep([control_pts[:, 1], control_pts[:, 0]], s=0)
                unew = np.arange(0, 1.01, 0.01)
                new_points = splev(unew, tck)
                return np.vstack(new_points).T

            smooth_kpts = continuous_kpts(np.vstack((kpts, kpts[0])))  # Adding the first keypoint to close the path
            codes = [Path.MOVETO] + [Path.CURVE4] * (len(smooth_kpts) - 2) + [Path.CLOSEPOLY]
            p = Path(smooth_kpts, codes)

            h, w = dp.visible_spectrum.shape[:2]
            xx, yy = np.meshgrid(range(h), range(w))
            img_pts = np.vstack([xx.ravel(), yy.ravel()]).T
            mask = p.contains_points(img_pts)
            mask = mask.reshape(dp.visible_spectrum.shape[:2])
            return mask

        dense_kpts = dp.keypoints.face.dense.coords_2d
        left_eye_keypoints = dense_kpts[left_eye_upper_idx + left_eye_lower_idx]
        right_eye_keypoints = dense_kpts[right_eye_upper_idx + right_eye_lower_idx]
        height, width, _ = dp.infrared_spectrum.shape

        # Combining original eyeballs segmentation map with calculated keypoints segmaps for the areas behind glasses
        segmap = dp.semantic_segmentation
        keypoints_based_mask = mask_from_kpts(left_eye_keypoints) | mask_from_kpts(right_eye_keypoints)
        glasses_mask = get_children_masks(segmap, dp.semantic_segmentation_metadata.glasses)
        keypoints_based_mask &= glasses_mask

        eye_cmap = dp.semantic_segmentation_metadata.human.head.eye
        segmentation_based_mask = (segmap == eye_cmap.left.eyeball) | (segmap == eye_cmap.right.eyeball)
        segmentation_based_mask = np.all(segmentation_based_mask, axis=2)
        return keypoints_based_mask | segmentation_based_mask

    # Fits an ellipse to the keypoints and returns a binary mask
    def get_ellipse_mask(ellipse_keypoints, height, width):
        yy, xx = np.meshgrid(range(height), range(width))
        ell = EllipseModel()
        ell.estimate(ellipse_keypoints)
        if ell.params is None:
            return np.zeros(xx.shape, dtype=bool)
        xc, yc, a, b, theta = ell.params
        theta = np.radians(theta)
        in_ellipse = ((((xx - xc) * np.sin(theta) + (yy - yc) * np.cos(theta)) / a) ** 2
                      + (((xx - xc) * np.cos(theta) + (yy - yc) * np.sin(theta)) / b) ** 2) < 1
        return in_ellipse
    h, w, _ = dp.infrared_spectrum.shape

    irises_keypoints = dp.actor_metadata.iris_circle.coords_2d[0]
    pupils_keypoints = dp.actor_metadata.pupil_circle.coords_2d[0]

    right_iris_mask = get_ellipse_mask(irises_keypoints.right_eye, h, w)
    left_iris_mask = get_ellipse_mask(irises_keypoints.left_eye, h, w)
    right_pupil_mask = get_ellipse_mask(pupils_keypoints.right_eye, h, w)
    left_pupil_mask = get_ellipse_mask(pupils_keypoints.left_eye, h, w)

    right_iris_mask[right_pupil_mask] = False
    left_iris_mask[left_pupil_mask] = False

    eyeballs = eyeball_mask(dp)
    irises = left_iris_mask | right_iris_mask
    pupils = left_pupil_mask | right_pupil_mask

    irises &= eyeballs
    pupils &= eyeballs

    return eyeballs, irises, pupils


def reconstruct_3d(dp):
    h, w, _ = dp.infrared_spectrum.shape
    yy, xx = np.meshgrid(range(h), range(w), indexing='ij')
    zz = dp.depth[..., 0]
    # Since the Z grid comes from the depth map, it is already in camera coordinates.
    # The X and Y grids are in image coordinates and have to be transformed to camera coordinates.
    # We will use the intrinsic matrix and homogeneous coordinates to move from image coordinates to camera coordinates:
    pixels_position_homogeneous = np.stack([xx * zz, yy * zz, zz], axis=2)
    pixels_position_homogeneous = pixels_position_homogeneous.reshape(-1, 3).T
    pixels_position_camera = (np.linalg.inv(dp.camera_metadata.intrinsic_matrix) @ pixels_position_homogeneous).T
    pixels_position_camera = pixels_position_camera.reshape(h, w, 3)
    return pixels_position_camera


# Preprocessing and normalization according to https://docs.datagen.tech/en/latest/Modalities/normal.html
def preprocessed_normals(dp):
    normal_maps = dp.normal_maps.copy()
    normal_maps = 2 * normal_maps - 1
    normal_maps[..., 1:] *= -1
    norm = np.linalg.norm(normal_maps, axis=-1)
    norm[norm == 0] = 1
    normal_maps /= norm[..., None]
    return normal_maps


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


# Takes a specific field in the colormap and extracts a list of all its children segments' colors
def get_children_masks(segmap, cmap_entry):
    if cmap_entry is None:
        return np.zeros(segmap.shape[:2], dtype=bool)
    elif type(cmap_entry) == np.ndarray:
        return (segmap == cmap_entry).all(axis=-1)
    elif type(cmap_entry) == datagen.modalities.textual.base.segmentation.ColoredSegment:
        return (segmap == cmap_entry.color).all(axis=-1)

    seg_mask = np.zeros(segmap.shape[:2], dtype=bool)
    for sub_segment in cmap_entry.sub_segments:
        seg_mask |= get_children_masks(segmap, sub_segment)
    return seg_mask


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
