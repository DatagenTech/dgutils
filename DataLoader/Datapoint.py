import numpy as np
from numpy import ndarray
from dataclasses import dataclass
from collections import namedtuple as Structure

# Generated using
# for key, value in dp.__dict__.items():
# print('{} : {}'.format(key, type(value).__name__))
# In DataPointHandler load function

@dataclass(frozen=True)
class Datapoint:
    # Actor metadata
    face_expression: Structure
    facial_hair_included: bool
    identity_label: Structure
    identity_id: str
    head_metadata: Structure
    apex_of_cornea_point: Structure
    center_of_rotation_point: Structure
    iris_circle: Structure
    center_of_iris_point: Structure
    pupil_circle: Structure
    center_of_pupil_point: Structure
    eye_gaze: Structure

    # Camera metadata
    camera_name: str
    camera_type: str
    location: ndarray
    orientation: Structure
    aspect_px: ndarray
    resolution_px: ndarray
    fov: Structure
    sensor: Structure
    intrinsic_matrix: ndarray
    extrinsic_matrix: ndarray

    # Standard keypoints
    keypoints_3d_coordinates: ndarray
    keypoints_2d_coordinates: ndarray
    is_visible: ndarray

    # Dense keypoints
    dense_keypoints_3d_coordinates: ndarray
    dense_keypoints_2d_coordinates: ndarray
    dense_is_visible: ndarray

    # Face bounding box
    min_x: int
    min_y: int
    max_x: int
    max_y: int

    # Semantic segmentation color map
    semantic_seg_colormap: Structure

    # Image modalities handlers. This is a private field.
    # Use the property fields below to access image modalities
    image_handlers: Structure

    # Image modalities getters. Uses lazy loading.
    @property
    def depth_img(self):
        return self.image_handlers.depth_img.img

    @property
    def rgb_img(self):
        return self.image_handlers.rgb_img.img

    @property
    def normals_map(self):
        return self.image_handlers.normals_map.img

    @property
    def semantic_seg_map(self):
        return self.image_handlers.semantic_seg_map.img


