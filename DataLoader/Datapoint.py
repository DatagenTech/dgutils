import numpy as np
from numpy import ndarray
from dataclasses import dataclass
from collections import namedtuple as Structure

# Generated using
#for key, value in dp.__dict__.items():
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

    # Image modalities handlers. This is a private field.
    # Use the property fields below to access image modalities
    image_handlers: Structure

    # Image modalities getters. Uses lazy loading.
    @property
    def depth_img(self):
        return self.image_handlers.depth_img.img

    @property
    def ir_img(self):
        return self.image_handlers.ir_img.img

    @property
    def rgb_img(self):
        return self.image_handlers.rgb_img.img

    @property
    def normals_map(self):
        return self.image_handlers.normals_map.img

    @property
    def semantic_seg_map(self):
        return self.image_handlers.semantic_seg_map.img


    # TODO: Remove the struct_only comparison and create a real unit test
    def __repr__(self) -> str:
        return self.repr_aux(struct_only=False)

    def repr_aux(self, struct_only : bool):
        """
        Gives a representation of the data point
        :param struct_only: Print only the structure of the Datapoint. Useful for integrity check
        :return: A printable string
        """
        def repr_dict(obj, depth):
            rep = ''
            with np.printoptions(threshold=0, precision=3, edgeitems=2):
                # Variable number of cameras makes the structure variable
                if 'camera_1' in obj and struct_only:
                    obj = {'camera': obj['camera_1']}
                for key, value in iter(sorted(obj.items())):
                    if self.__isinstance_namedtuple(value):
                        value = repr_dict(value._asdict(), depth + 1)
                    elif isinstance(value, np.ndarray):
                        if struct_only:
                            value = f'np.ndarray ({len(value.shape)} dimensional)'
                        elif len(value.shape) > 1:
                            value = f'np.ndarray (shape = {value.shape})'
                    else:
                        if struct_only:
                            value = str(type(value))

                    content_str = f'{key} : {value} '
                    if depth == 0:
                        rep += content_str + '\n'
                    else:
                        rep += '{' + content_str + '}'
            return rep
        return repr_dict(self.__dict__, depth=0)

    def __eq__(self, other):
        """
        Compare the data point with another one
        :param other: The data point object to be comapred with
        :param comparison: Whether to compare the structures of the data points or the values themselves
        :return: True if the Datapoints are equal, False otherwise
        """
        def recursive_compare(d1 : dict, d2 : dict):
            # Check that both dictionaries have the same set of keys
            if d1.keys() != d2.keys():
                return False

            for k in d1.keys():
                v1, v2 = d1[k], d2[k]
                # There should not be any dictionary inside
                assert not isinstance(v1, dict) and not isinstance(v2, dict)
                # Check that values have the same type for each key
                if type(v1) != type(v2):
                    return False
                # If dictionaries, check recursively
                elif self.__isinstance_namedtuple(v1):
                    if not recursive_compare(v1._asdict(), v2._asdict()):
                        return False
                # If not, compare the fields against each other
                else:
                    # Value comparison
                    if type(v1) == np.ndarray:
                        if (v1 != v2).any():
                            return False
                    else:
                        if v1 != v2:
                            return False
            return True

        return recursive_compare(self.__dict__, other.__dict__)


    @staticmethod
    def __isinstance_namedtuple(obj) -> bool:
        """
        Check if obj is a namedtuple
        """
        return (
                isinstance(obj, tuple) and
                hasattr(obj, '_asdict') and
                hasattr(obj, '_fields')
        )

