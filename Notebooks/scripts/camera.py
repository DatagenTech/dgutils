from collections import namedtuple

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from scipy.spatial.transform import Rotation


@dataclass
class FocusInfo:
    focal_length: float
    field_of_view: float

    @classmethod
    def from_field_of_view(cls, field_of_view: float, sensor_width: float) -> "FocusInfo":
        focal_length = sensor_width / (np.tan(field_of_view / 2) * 2)
        return cls(field_of_view=field_of_view, focal_length=focal_length)

    @classmethod
    def from_focal_length(cls, focal_length: float, sensor_width: float) -> "FocusInfo":
        field_of_view = 2 * np.arctan2(0.5 * sensor_width, focal_length)
        return cls(field_of_view=field_of_view, focal_length=focal_length)


@dataclass
class SensorShape:
    width: float
    height: float


class Intrinsics:
    def __init__(
        self, focus_info: FocusInfo, resolution: Tuple[int, int], sensor_shape: SensorShape, pixel_aspect_ratio: float,
    ):
        self.focus_info = focus_info
        self.resolution = resolution
        self.sensor_shape = sensor_shape
        self.pixel_aspect_ratio = pixel_aspect_ratio
        self.skew = 0
        self._matrix = self._get_matrix()

    def _get_matrix(self):
        pixels_per_mm_height = self.resolution[0] / self.sensor_shape.height
        f_x = self.focus_info.focal_length * pixels_per_mm_height
        pixels_per_mm_width = self.resolution[1] / self.sensor_shape.width
        f_y = self.focus_info.focal_length * pixels_per_mm_width
        return np.array(
            [
                [f_x, self.skew, self.resolution[0] / 2],
                [0.0, f_y * self.pixel_aspect_ratio, self.resolution[1] / 2],
                [0.0, 0.0, 1.0],
            ]
        )

    @property
    def matrix(self):
        return self._matrix


class CameraRotation:
    DEFAULT_ROTATION = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    def __init__(self, rotation: np.ndarray):
        self.matrix = rotation

    @classmethod
    def from_euler(cls, yaw: float, pitch: float, roll: float, degrees: bool = True):
        matrix = Rotation.from_euler("yxz", (-yaw, -pitch, roll), degrees=degrees).as_matrix()
        return cls(matrix @ cls.DEFAULT_ROTATION)


class Extrinsics:
    def __init__(self, rotation: CameraRotation, location: np.ndarray):
        self.rotation = rotation
        self.location = location
        self.matrix = self._get_matrix(rotation, location)

    def _get_matrix(self, rotation: CameraRotation, location: np.ndarray) -> np.ndarray:
        rotated_location = -(rotation.matrix @ location)
        extrinsic = np.hstack([rotation.matrix, rotated_location[..., np.newaxis]])
        return extrinsic


@dataclass
class Camera:
    intrinsics: Intrinsics
    extrinsics: Extrinsics

    def coordinates_to_pixel_space(self, coordinates):
        homogenous_coordinates = np.hstack([coordinates, np.ones((len(coordinates), 1))]).T  # 4 x N
        in_camera_frame = self.extrinsics.matrix @ homogenous_coordinates  # 3 x N
        in_pixel_frame = self.intrinsics.matrix @ in_camera_frame  # 3 x N
        in_pixel_space = in_pixel_frame[:2] / in_pixel_frame[2, np.newaxis]  # 2 x N
        return in_pixel_space.T  # N x 2

    def get_pixels_in_frame(self, pixels: np.ndarray) -> int:
        x, y = pixels.T
        image_height, image_width = self.intrinsics.resolution
        in_height = (0 <= x) * (x < image_height)
        in_width = (0 <= y) * (y < image_width)
        in_frame = in_height * in_width
        return in_frame
