import copy
import json
import os
from os.path import join, dirname
import cv2
import numpy as np
from Datapoint import Datapoint
from ImageHandler import ImageHandler
from structure import structure
import difflib


class DatapointHandler:
    DP_STRUCT_FILE = join(dirname(__file__), 'dp_struct.txt')
    def __init__(self, path, check_integrity=True):
        """
        :param path: The path for the data point folder
        :param check_integrity: Check if the datapoint is structured identically as the template datapoint
        """
        self._path = path
        self._loaded = False
        self._dp = None
        self._check_integrity = check_integrity
        if self._check_integrity:
            with open(self.DP_STRUCT_FILE, 'r') as file:
                self.dp_struct = file.read()

    @property
    def dp(self):
        # Load the datapoint if needed
        self.__load()
        return self._dp

    def __load(self):
        """
        Loads a datapoint to memory
        :return: None
        """
        # Lazy loading
        if self._loaded:
            return

        if not self.check_structure(self._path):
            raise ValueError(f'The directory {self._path} does not follow the data point standard structure. Please check that no '
                               'file or folder was removed or renamed.')
        members = self.__load_json_files()
        members.update(self.__load_imgs())
        # Turning all the internal fields to namedtuples when possible,
        # and converting back the external structure to be a dictionary
        members = structure(members)._asdict()

        # Selecting only the fields that are in datapoints and ignoring additional fields
        datapoint_fields = Datapoint.__annotations__.keys()
        members_in_datapoint = {key: members[key] for key in datapoint_fields}
        self._dp = Datapoint(**members_in_datapoint)

        # Integrity checking
        if self._check_integrity:
            current_rep = self._dp.repr_aux(struct_only=True)
            baseline_repr = self.dp_struct
            if current_rep.replace(" ", "") != baseline_repr.replace(" ", ""):
                diff = ''
                for text in difflib.unified_diff(current_rep.split("\n"), baseline_repr.split("\n")):
                    if text[:3] not in ('+++', '---', '@@ '):
                        diff += text + '\n'
                raise ValueError(f"The datapoint does not follow the standard format. Please check its integrity\n"
                                 f"Diff output:\n"
                                 f"{diff}")

        self._loaded = True

    @classmethod
    def check_structure(cls, path) -> bool:
        """
        Checks that a path is structured as a data point
        :param path: The datapoint path
        """
        for fn in cls.__get_filelist():
            if not os.path.exists(join(path, fn)):
                return False
        return True

    def __load_json_files(self) -> dict:
        """
        Loads all the data point JSON files
        :return: The loaded data
        """
        def standardize_kpts(kpts_dict, first_idx):
            return np.array([kpts_dict[str(i)] for i in range(first_idx, len(kpts_dict) + first_idx)])

        all_dict = {}
        for fn in self.__get_filelist():
            if not fn.endswith('.json'):
                continue
            with open(join(self._path, fn)) as json_file:
                local_dict = json.load(json_file)
                if 'dense_keypoints' in fn:
                    keys = list(local_dict.keys())
                    for key in keys:
                        local_dict['dense_' + key] = local_dict.pop(key)
                # Converts lists to numpy arrays
                for key in local_dict:
                    if isinstance(local_dict[key], list):
                        local_dict[key] = np.array(local_dict[key])

                # Sanity test (for debug). Check that we don't erase a key that is in both dictionaries
                shared_items = {k: all_dict[k] for k in all_dict if
                                k in local_dict and all_dict[k] == local_dict[k]}
                assert len(shared_items) == 0

                all_dict.update(local_dict)

        all_dict = self.__nested_coords_to_numpy(all_dict)
        #FIXME: Make it shorter and cleaner
        for dense in [False, True]:
            for key in ['keypoints_2d_coordinates', 'keypoints_3d_coordinates', 'is_visible']:
                if dense:
                    key = 'dense_' + key
                all_dict[key] = standardize_kpts(all_dict[key], first_idx=0 if dense else 1)

        return all_dict

    def __load_imgs(self) -> dict:
        """
        Loads all the image files
        :return: The loaded data
        """

        img_dict = {'depth_img' : 'depth.exr', 'ir_img' : 'infrared_spectrum.png', 'rgb_img':
                    'visible_spectrum.png', 'normals_map': 'normal_maps.exr',
                    'semantic_seg_map':'semantic_segmentation.exr'}
        for k, v in img_dict.items():
            img_dict[k] = join(self._path, 'camera', v)

        exr_open = lambda img_path : cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        bgr2rgb = lambda img : cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        reduce_channels = lambda img : img.mean(axis=-1)
        img_dict['depth_img'] = ImageHandler(img_dict['depth_img'], [exr_open, reduce_channels])
        img_dict['ir_img'] = ImageHandler(img_dict['ir_img'], [cv2.imread, bgr2rgb])
        img_dict['rgb_img'] = ImageHandler(img_dict['rgb_img'], [cv2.imread, bgr2rgb])
        img_dict['normals_map'] = ImageHandler(img_dict['normals_map'], [exr_open, bgr2rgb])
        img_dict['semantic_seg_map'] = ImageHandler(img_dict['semantic_seg_map'], [exr_open, bgr2rgb])

        return {'image_handlers' : img_dict}

    @classmethod
    def __nested_coords_to_numpy(cls, obj):
        """
        Iterates over all nested dictionaries and converts inner xyz and yaw/pitch/roll dictionaries to numpy arrays
        :param obj: A list or dictionary of the object to iterate on
        :return: The same object structure with dicts/lists converted to Numpy arrays where applicable
        """
        def coords_to_numpy(d: dict) -> np.ndarray:
            if len(d) == 2:
                return np.array([d['x'], d['y']])
            elif len(d) == 3:
                if 'x' in d.keys():
                    return np.array([d['x'], d['y'], d['z']])
                elif 'yaw' in d.keys():
                    return np.array([d['yaw'], d['pitch'], d['roll']])

            # If we didn't return anything, the format is not correct
            raise ValueError("Invalid dictionary format")

        def is_coord_dict(d: dict) -> bool:
            if not isinstance(d, dict):
                return False
            numeric_values = all(isinstance(v, (int, float)) for v in d.values())
            coords_2d = (len(d) == 2 and all(k in d for k in ('x', 'y')))
            coords_3d = (len(d) == 3 and all(k in d for k in ('x', 'y', 'z')))
            coords_rotation = (len(d) == 3 and all(k in d for k in ('yaw', 'pitch', 'roll')))
            correct_keys = (coords_2d or coords_3d or coords_rotation)
            return numeric_values and correct_keys

        obj_new = copy.deepcopy(obj)
        if is_coord_dict(obj_new):
            return coords_to_numpy(obj_new)

        if isinstance(obj_new, list):
            for i in range(len(obj_new)):
                obj_new[i] = cls.__nested_coords_to_numpy(obj_new[i])
            if all(isinstance(item, np.ndarray) for item in obj_new):
                obj_new = np.stack(obj_new, axis=0)
        elif isinstance(obj_new, dict):
            keys = obj_new.keys()
            for key in keys:
                obj_new[key] = cls.__nested_coords_to_numpy(obj_new[key])
        return obj_new

    @staticmethod
    # All the files are given except semantic_segmentation_metadata.json
    def __get_filelist():
        filelist = ['actor_metadata.json']
        filelist += [join('camera', fn) for fn in
                     'camera_metadata.json dense_keypoints.json depth.exr environment.json face_bounding_box.json ' \
                     'visible_spectrum.png infrared_spectrum.png normal_maps.exr semantic_segmentation.exr ' \
                     'standard_keypoints.json'.split()]
        return filelist
