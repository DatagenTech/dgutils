import copy
import json
import os
from glob import glob
from os.path import join, dirname
import cv2
import numpy as np
from Datapoint import Datapoint
from ImageHandler import ImageHandler
from structure import structure
import difflib


class DatapointHandler:
    def __init__(self, render_path):
        """
        :param render_path: The path of the visible_spectrum file
        """
        self._render_path = render_path
        self._loaded = False
        self._dp = None

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

        if not self.check_structure(self._render_path):
            raise ValueError(f'The directory of {self._render_path} does not follow the data point standard structure. Please check that no '
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
        self._loaded = True


    @classmethod
    def check_structure(cls, render_path) -> bool:
        """
        Checks that a path is structured as a data point
        :param render_path: The datapoint's visible_spectrum file path
        """
        if not os.path.isfile(render_path):
            return False
        # Checks if all the visual/JSON files are present in the directory
        for fn in cls.__get_filelist(render_path):
            if not os.path.isfile(join(dirname(render_path), fn)):
                return False
        return True

    @staticmethod
    def __standardize_seg_color(color):
        return np.round(np.asarray(color).astype(np.float16), 2).astype(np.float32)

    def __load_json_files(self) -> dict:
        """
        Loads all the data point JSON files
        :return: The loaded data
        """

        all_dict = {}
        for fn in self.__get_filelist(self._render_path):
            if not fn.endswith('.json'):
                continue
            with open(join(dirname(self._render_path), fn)) as json_file:
                local_dict = json.load(json_file)
                if 'dense_keypoints' in fn:
                    keys = list(local_dict.keys())
                    for key in keys:
                        local_dict['dense_' + key] = local_dict.pop(key)
                if 'semantic_segmentation_metadata' in fn:
                    if 'human' not in local_dict:
                        #Old version of the segmentation map
                        local_dict = {key: self.__standardize_seg_color(val) for key, val in local_dict.items()}
                    local_dict = {'semantic_seg_colormap': local_dict}

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
                # Transforms the keypoint numbered list ("0": [], "1": []...) to a Numpy array
                all_dict[key] = np.array(list(all_dict[key].values()))

        return all_dict

    def __load_imgs(self) -> dict:
        """
        Loads all the image files
        :return: The loaded data
        """
        img_dict = {'depth_img' : 'depth.exr', 'rgb_img': os.path.basename(self._render_path),
                    'normals_map': 'normal_maps.exr',
                    'semantic_seg_map':'semantic_segmentation.png'}

        for k, v in img_dict.items():
            img_dict[k] = join(dirname(self._render_path), v)

        exr_open = lambda img_path : cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        bgr2rgb = lambda img : cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        reduce_channels = lambda img : img.mean(axis=-1)
        img_dict['depth_img'] = ImageHandler(img_dict['depth_img'], [exr_open, reduce_channels])
        img_dict['rgb_img'] = ImageHandler(img_dict['rgb_img'], [cv2.imread, bgr2rgb])
        img_dict['normals_map'] = ImageHandler(img_dict['normals_map'], [exr_open, bgr2rgb])
        img_dict['semantic_seg_map'] = ImageHandler(img_dict['semantic_seg_map'], [exr_open, bgr2rgb, self.__standardize_seg_color])

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
                if {'x', 'y', 'z'} == set(d.keys()):
                    return np.array([d['x'], d['y'], d['z']])
                elif {'yaw', 'pitch', 'roll'} == set(d.keys()):
                    return np.array([d['yaw'], d['pitch'], d['roll']])
                elif {'R', 'G', 'B'} == set(d.keys()):
                    return np.array([d['R'], d['G'], d['B']])

            # If we didn't return anything, the format is not correct
            raise ValueError("Invalid dictionary format")

        def is_coord_dict(d: dict) -> bool:
            if not isinstance(d, dict):
                return False
            numeric_values = all(isinstance(v, (int, float)) for v in d.values())
            coords_2d = {'x', 'y'} == set(d.keys())
            coords_3d = {'x', 'y', 'z'} == set(d.keys())
            coords_rotation = {'yaw', 'pitch', 'roll'} == set(d.keys())
            coords_RGB = {'R', 'G', 'B'} == set(d.keys())
            correct_keys = (coords_2d or coords_3d or coords_rotation or coords_RGB)
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
    # All the files are given except semantic_segmentation_metadata.json and envrionment.json
    # The files are defined in relation to the render_path
    # visible_spectrum is not included as it is defined in self._render_path
    def __get_filelist(render_path):
        filelist = '../actor_metadata.json ../semantic_segmentation_metadata.json camera_metadata.json ' \
                   'dense_keypoints.json depth.exr face_bounding_box.json normal_maps.exr standard_keypoints.json'.split()

        segmentation_map_path = glob(join(dirname(render_path), 'semantic_segmentation.???'))
        if not segmentation_map_path:
            raise RuntimeError('No segmentation file found')
        if len(segmentation_map_path) > 1:
            raise RuntimeError('Multiple segmentation files')

        filelist += [os.path.basename(segmentation_map_path[0])]

        return filelist
