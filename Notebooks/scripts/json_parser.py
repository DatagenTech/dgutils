import numpy as np
from scripts.camera import Intrinsics, FocusInfo, SensorShape, CameraRotation, Extrinsics, Camera


class CameraJSONParser:
    def get_camera(self, camera):
        intrinsics = self.get_intrinsics(camera)
        extrinsics = self.get_extrinsics(camera)
        return Camera(intrinsics=intrinsics, extrinsics=extrinsics)

    def get_intrinsics(self, camera):
        focus_info = self.get_focus_info(camera)
        sensor_shape = self.get_sensor_shape(camera)
        resolution = self.get_resolution(camera)
        return Intrinsics(
            focus_info=focus_info, resolution=resolution, pixel_aspect_ratio=1.0, sensor_shape=sensor_shape,
        )

    def get_extrinsics(self, camera):
        rotation = self.get_rotation(camera)
        location = self.get_location(camera)
        return Extrinsics(rotation=rotation, location=location)

    def get_resolution(self, camera):
        return camera["resolution_px"]["height"], camera["resolution_px"]["width"]

    def get_focus_info(self, camera):
        return FocusInfo.from_field_of_view(camera["fov"]["horizontal"] * np.pi / 180, camera["sensor"]["width"])

    def get_sensor_shape(self, camera):
        return SensorShape(height=camera["sensor"]["height"], width=camera["sensor"]["width"])

    def get_rotation(self, camera):
        rotation = camera["orientation"]["rotation"]
        return CameraRotation.from_euler(rotation["yaw"], rotation["pitch"], rotation["roll"], degrees=True)

    def get_location(self, camera):
        l = camera["location"]
        return np.array([l["x"], l["y"], l["z"]])
