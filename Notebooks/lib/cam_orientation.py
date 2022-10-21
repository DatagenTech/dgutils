import math
import numpy as np
from datagen.api.datapoint import assets

# Insert the camera's and the head's location values
CAM_X = -2.36
CAM_Y = -2.5
CAM_Z = -1.54

HEAD_X = 1.2
HEAD_Y = -0.85
HEAD_Z = -2.3


def calc_cam_rotation(cam_location: assets.Point, head_location: assets.Point):
    # Our camera has a 0.12 buffer at the z value
    x = cam_location.x - head_location.x
    y = cam_location.y - head_location.y
    z = cam_location.z - 0.12 - head_location.z

    # Find the heading
    yaw_rad = math.atan2(x, y)
    yaw = 180 - ((yaw_rad * 180) / math.pi)
    if yaw > 180:
        yaw -= 360
    if yaw < -180:
        yaw += 360

    # Find the pitch
    pitch_rad = math.atan2(z, math.sqrt(x ** 2 + y ** 2))
    pitch = -((pitch_rad * 180) / math.pi)

    roll = 0.0  # It's a pinhole camera

    camera_rotation = assets.CameraRotation(yaw=yaw, pitch=pitch, roll=roll)

    return camera_rotation


def get_n_cameras_ring(num_of_cameras: int, head_location: assets.Point, radius: float, z_value: float, wavelength: str,
                       projection_type: assets.Projection = assets.Projection.PERSPECTIVE,
                       res_width: int = 512, res_height: int = 512,
                       fov_horiz: float = 9.0, fov_vert: float = 9.0,
                       sensor_width: float = 34):

    cameras = []

    if wavelength == "nir":
        wavelength = assets.Wavelength.NIR
    else:
        wavelength = assets.Wavelength.VISIBLE

    for i, xy_angle in enumerate(np.linspace(0, 360, num=num_of_cameras, endpoint=False)):
        xy_rad = (xy_angle * math.pi) / 180
        x = (math.sin(xy_rad) + head_location.x) * radius
        y = (math.cos(xy_rad) + head_location.y) * radius
        z = z_value + 0.12

        cam_location = assets.Point(x=x, y=y, z=z)
        cam_rotation = calc_cam_rotation(cam_location, head_location)

        camera = assets.Camera(
            name="Camera" + str(i),
            intrinsic_params=assets.IntrinsicParams(
                projection=projection_type,
                resolution_width=res_width,
                resolution_height=res_height,
                fov_horizontal=fov_horiz,
                fov_vertical=fov_vert,
                sensor_width=sensor_width,
                wavelength=wavelength
            ),
            extrinsic_params=assets.ExtrinsicParams(
                location=cam_location,
                rotation=cam_rotation
            )
        )

        cameras.append(camera)

    return


if __name__ == '__main__':
    # cam_loc = assets.Point(x=CAM_X, y=CAM_Y, z=CAM_Z)
    head_loc = assets.Point(x=HEAD_X, y=HEAD_Y, z=HEAD_Z)
    #
    # cam_rot = calc_cam_rotation(cam_loc, head_loc)
    #
    # print(f"Camera Yaw: {cam_rot.yaw} degrees\nCamera Pitch: {cam_rot.pitch} degrees\nCamera Roll: {cam_rot.roll} degrees")

    get_n_cameras_ring(num_of_cameras=14, head_location=head_loc, radius=1.8, z_value=2, wavelength="visible")