import datagen as dg
import numpy as np
import argparse
import json
import cv2
import os
from nested_lookup import nested_lookup, nested_alter


"""
This script applies a fish-eye distortion to a whole dataset:
    - Warps all the images within a dataset (RGB, IR, segmentation, depth, normals) and saves them as
      <original_name_fisheye.png/.exr> in the same folder or overwrites the original ones if indicated.
    - Same for the key points - each JSON has its fisheye version: <original_name_fisheye.json> (if not overwritten)
    - There is an option to crop the images to fit their new size after warping
    - After cropping, there is an option to upscale the images back to their original size
    - There is an option to overwrite all the modalities in order to be able to use the DG SDK after warping a dataset
    - There is an option to output a video for each scene in order to visualize the distorted frames and the key points

Usage:
In your terminal, run: python fish_eye.py <path_to_dataset_folder>
Optional arguments:
    -d - A list of the 4 distortion coefficients: k1, k2, k3, k4 
        (default is: [-0.11620065995544242, 0.2445535080942127, -0.08597530941689592, 0.00397530941689592]) - 
        According to OpenCV's fisheye model documentation
    -c - Crops the distorted image (default is not to crop)
    -u - Upscales the cropped image back to the original size (default is not to upscale)
    -o - Overwrites the original modalities with the fisheye version (default is not to overwrite)
    -v - Creates a video in the scene's folder of the distorted frames and key points (default is not to create a video)
"""


def create_pixels_maps(im_pixels_map, width, height, cam_intrinsic, dist_coeff):
    # Undistorted pixels map - (H*W, 1, 2)
    undistorted_pixel_map = cv2.fisheye.undistortPoints(im_pixels_map, cam_intrinsic, dist_coeff)
    # Convert to homogeneous coordinates - (H*W, 1, 3)
    undistorted_pixel_map = np.dstack((undistorted_pixel_map, np.ones((width * height, 1))))
    # Dot-product --> camera coordinates - (H*W, 1, 3)
    undistorted_pixel_map = np.tensordot(undistorted_pixel_map, cam_intrinsic, axes=(2, 1))
    # Back to non-homogeneous coordinates - (H*W, 1, 2)
    undistorted_pixel_map = undistorted_pixel_map[:, :, :-1] / undistorted_pixel_map[:, :, -1][:, np.newaxis]
    undistorted_pixel_map = undistorted_pixel_map.reshape((height, width, 2))  # Reshaping to: (H, W, 2)
    undistorted_pixel_map = np.flip(undistorted_pixel_map, axis=2)  # Since cv2 is height first

    # Distorted pixels to undistorted pixels
    # Convert to homogeneous coordinates - (H*W, 1, 3)
    distorted_pixel_map = np.dstack((im_pixels_map, np.ones((width * height, 1))))
    camera_intrinsic_inv = np.linalg.inv(cam_intrinsic)
    # Dot-product --> camera coordinates - (H*W, 1, 3)
    distorted_pixel_map = np.tensordot(distorted_pixel_map, camera_intrinsic_inv, axes=(2, 1))
    # Back to non-homogeneous coordinates - (H*W, 1, 2)
    distorted_pixel_map = distorted_pixel_map[:, :, :-1] / distorted_pixel_map[:, :, -1][:, np.newaxis]
    # Distorted pixels map - (W*H, 1, 2)
    distorted_pixel_map = cv2.fisheye.distortPoints(distorted_pixel_map, cam_intrinsic, dist_coeff)
    distorted_pixel_map = distorted_pixel_map.reshape((height, width, 2))  # Reshaping to: (H, W, 2)

    return undistorted_pixel_map, distorted_pixel_map


def lookup_callback(point_dict, distorted_map, orig_im_map, x_red, y_red, ratio_x, ratio_y):
    point = np.array([point_dict["x"], point_dict["y"]])
    new_kpt_coord = np.array([0, 0])
    try:
        new_point = distorted_map[orig_im_map[round(point[1])][round(point[0])][0]][orig_im_map[round(point[1])][round(point[0])][1]]
        new_kpt_coord[0] = round((new_point[1] - x_red) * ratio_x)
        new_kpt_coord[1] = round((new_point[0] - y_red) * ratio_y)
        new_kpt_coord = new_kpt_coord.astype(int)
    except:
        # If a keypoint is not visible
        new_kpt_coord = np.array([None, None])  # TODO: extend to invisible keypoints
    return new_kpt_coord.tolist()


def create_distorted_jsons(dp_cam_path, dis_pixel_map, orig_pixel_map, x_crop, y_crop, ar_x, ar_y, visualize, overwrite):
    all_dist_jsons = {}
    kpts_dir = os.path.join(dp_cam_path, "key_points")
    for j_file in os.listdir(kpts_dir):
        if j_file.endswith(".json") and "fisheye" not in j_file:
            j_file_path = os.path.join(kpts_dir, j_file)
            with open(j_file_path, "r") as f:
                data = json.load(f)
                new_dict = nested_alter(document=data, key="pixel_2d", callback_function=lookup_callback,
                                        function_parameters=[dis_pixel_map, orig_pixel_map,
                                                             x_crop, y_crop,
                                                             ar_x, ar_y])
            if overwrite:
                fisheye_json = open(j_file_path, "w")
            else:
                fisheye_json = open(j_file_path.replace(".json", "_fisheye.json"), "w")
            json.dump(new_dict, fisheye_json, indent=2)
            fisheye_json.close()
            if visualize:
                all_dist_jsons[j_file] = new_dict

    return all_dist_jsons


def apply_fisheye(data_point, distortion_coefficients, crop, upscale, overwrite, visualize):
    # Load the images and camera's intrinsic matrix
    try:
        visible_spectrum_image = cv2.cvtColor(data_point.visible_spectrum, cv2.COLOR_RGB2BGR)
    except:
        visible_spectrum_image = None
    try:
        nir_image = cv2.cvtColor(data_point.infrared_spectrum, cv2.COLOR_RGB2BGR)
    except:
        nir_image = None
    semantic_segmentation_image = cv2.cvtColor(data_point.semantic_segmentation, cv2.COLOR_RGB2BGR)
    depth_image = cv2.cvtColor(data_point.depth, cv2.COLOR_RGB2BGR)
    normal_maps_image = cv2.cvtColor(data_point.normal_maps, cv2.COLOR_RGB2BGR)
    camera_intrinsic = data_point.camera_metadata.intrinsic_matrix
    images = {
        data_point.visible_spectrum_image_name: visible_spectrum_image,
        "infrared_spectrum": nir_image,
        "semantic_segmentation": semantic_segmentation_image,
        "depth": depth_image,
        "normal_maps": normal_maps_image,
    }
    rgb_dist = None
    min_x = 0
    min_y = 0
    ratio_x = 1
    ratio_y = 1
    h, w, _ = semantic_segmentation_image.shape  # Assumes that all the images have the same size
    if len(distortion_coefficients) == 4:
        distortion_coefficients = np.array(distortion_coefficients)
    else:
        print("Invalid distortion coefficients! Should be an array of shape: (1, 4)")
        raise SystemExit

    # Image's pixels locations
    x_vec, y_vec = np.meshgrid(np.arange(w), np.arange(h))
    image_pixels_grid = np.stack((x_vec, y_vec), axis=2)
    image_pixels = image_pixels_grid.reshape((-1, 1, 2)).astype(np.float32)

    undistorted_map, distorted_map = create_pixels_maps(image_pixels, w, h, camera_intrinsic, distortion_coefficients)

    for image_name, image in images.items():
        if image is not None:
            h, w, ch = image.shape

            # visible_spectrum image --> Lanczos interpolation
            if image_name == data_point.visible_spectrum_image_name or image_name == "infrared_spectrum":
                interpolation_method = cv2.INTER_LANCZOS4
            # Segmentation mask/depth map/normals map --> nearest neighbor interpolation
            else:
                interpolation_method = cv2.INTER_NEAREST

            # Create a distorted image
            distorted_image = np.dstack(
                [
                    cv2.remap(
                        image[:, :, channel],
                        undistorted_map[:, :, 1].astype(np.float32),
                        undistorted_map[:, :, 0].astype(np.float32),
                        interpolation=interpolation_method,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    for channel in range(ch)
                ]
            )

            if crop:
                # Crop the distorted image
                min_y = np.ceil(distorted_map[int(h / 2), 0, 0]).astype(np.int32)
                max_y = np.ceil(distorted_map[int(h / 2), -1, 0]).astype(np.int32)
                min_x = np.ceil(distorted_map[0, int(w / 2), 1]).astype(np.int32)
                max_x = np.ceil(distorted_map[-1, int(w / 2), 1]).astype(np.int32)
                distorted_image = distorted_image[min_x:max_x, min_y:max_y]

                if upscale:
                    # Upscale the cropped image to its original size
                    cropped_im_h, cropped_im_w, _ = distorted_image.shape
                    ratio_x = h / cropped_im_h
                    ratio_y = w / cropped_im_w
                    distorted_image = cv2.resize(distorted_image, (w, h), interpolation=interpolation_method)

            # Save the images
            if image_name == data_point.visible_spectrum_image_name:
                if overwrite:
                    fe_im_name = image_name
                else:
                    fe_im_name = image_name.replace(".png", "_fisheye.png")
                cv2.imwrite(
                    os.path.join(data_point.camera_path, fe_im_name),
                    distorted_image,
                )
            if image_name == "semantic_segmentation" or image_name == "infrared_spectrum":
                if overwrite:
                    fe_im_name = ".png"
                else:
                    fe_im_name = "_fisheye.png"
                cv2.imwrite(
                    os.path.join(data_point.camera_path, image_name) + fe_im_name,
                    distorted_image,
                )
            elif image_name == "normal_maps" or image_name == "depth":
                if overwrite:
                    fe_im_name = ".exr"
                else:
                    fe_im_name = "_fisheye.exr"
                cv2.imwrite(
                    os.path.join(data_point.camera_path, image_name) + fe_im_name,
                    distorted_image,
                    [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF],
                )
            if visible_spectrum_image is not None and image_name == data_point.visible_spectrum_image_name:
                rgb_dist = distorted_image
            elif visible_spectrum_image is None and image_name == "infrared_spectrum":
                rgb_dist = distorted_image

    all_dist_kpts = create_distorted_jsons(data_point.camera_path,
                                           distorted_map,
                                           image_pixels_grid,
                                           min_x,
                                           min_y,
                                           ratio_x,
                                           ratio_y,
                                           visualize,
                                           overwrite)

    return rgb_dist, all_dist_kpts


def draw_keypoints(img, keypoints):
    for kpt in keypoints:
        if kpt[0] is not None and kpt[1] is not None:
            cv2.circle(img, (round(kpt[1]), round(kpt[0])), 0, (0, 0, 255), 3)
    return img


if __name__ == '__main__':
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Apply fish-eye distortion')
    parser.add_argument('dataset_path',
                        type=str,
                        help="Path to the dataset's directory which contains the scene_xxxxx or environment_xxxxx folder/s")
    parser.add_argument('-d',
                        '--distortion_coefficients',
                        type=float,
                        help="The 4 camera distortion coefficients ([k1, k2, k3, k4]) - a list of floats",
                        nargs='+',
                        default=[-0.11620065995544242, 0.2445535080942127, -0.08597530941689592, 0.00397530941689592])
    parser.add_argument('-c',
                        '--crop',
                        action='store_true',
                        help="Crops the distorted images")
    parser.add_argument('-u',
                        '--upscale',
                        action='store_true',
                        help="Upscales the cropped images")
    parser.add_argument('-o',
                        '--overwrite',
                        action='store_true',
                        help="Overwrites the original modalities")
    parser.add_argument('-v',
                        '--visualize',
                        action='store_true',
                        help="Creates a video in the scene's folder of the distorted frames and key points")
    args = parser.parse_args()

    # Video's parameters
    FPS = 5
    CODEC = 'MJPG'
    output_dist_image = None
    # Load the dataset
    try:
        dataset = dg.load(args.dataset_path)
    except ():
        print("Not a valid dataset directory!")
        raise SystemExit
    # Loop over the scenes
    print("\nProcessing...")
    for i, scene in enumerate(dataset.scenes):
        if args.visualize:
            vid_frame_size = scene.datapoints[0].camera_metadata.resolution_px.astype(int)
            video = cv2.VideoWriter(os.path.join(scene.path, 'distorted.avi'), cv2.VideoWriter_fourcc(*CODEC), FPS,
                                    vid_frame_size)
        # Loop over the datapoints (HIC --> frames, Faces --> Cameras)
        for j, dp in enumerate(scene.datapoints):
            # Applies fisheye distortion on all images and key points
            dist_image, dist_kpts = apply_fisheye(dp, args.distortion_coefficients, args.crop, args.upscale,
                                                  args.overwrite, args.visualize)
            if args.visualize:
                for json_name, json_kpts in dist_kpts.items():
                    coords = nested_lookup(key="pixel_2d", document=json_kpts)
                    all_dist_kpts = [[coord[0], coord[1]] for coord in coords]
                    output_dist_image = draw_keypoints(dist_image, all_dist_kpts)
                    if (output_dist_image.shape[:2] != vid_frame_size[::-1]).all():
                        output_dist_image = cv2.resize(output_dist_image, vid_frame_size, interpolation=cv2.INTER_LANCZOS4)
                video.write(output_dist_image)
        if args.visualize:
            video.release()

        print(
            f"Finished processing {j + 1} out of {len(scene.datapoints)} data-points in: {os.path.basename(scene.path)}")

    print(
        f"\nFinished processing {i + 1} out of {len(dataset.scenes)} scenes in: {os.path.basename(args.dataset_path)}\n")