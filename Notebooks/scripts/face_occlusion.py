from scipy.spatial import ConvexHull
import cv2
import numpy as np
import datagen as dg


# Define the dataset's path and the data-point that you're willing to check
DATASET_PATH = '../../resources/faces_2'
dp = dg.load(DATASET_PATH)[2]


def get_perc_vis_face(dp):
    face_dense_kpts_2d = dp.keypoints.face.dense.coords_2d
    image = dp.visible_spectrum
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = np.shape(image)
    seg_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    nose_colormap = np.array([dp.semantic_segmentation_metadata.human.head.nose.left,
                              dp.semantic_segmentation_metadata.human.head.nose.right])
    eyebrow_colormap = np.array([dp.semantic_segmentation_metadata.human.head.eyebrow.left,
                                 dp.semantic_segmentation_metadata.human.head.eyebrow.right])
    left_eye_colormap = np.array([dp.semantic_segmentation_metadata.human.head.eye.left.eyeball,
                                  dp.semantic_segmentation_metadata.human.head.eye.left.tear_duct,
                                  dp.semantic_segmentation_metadata.human.head.eye.left.eyelid])
    right_eye_colormap = np.array([dp.semantic_segmentation_metadata.human.head.eye.right.eyeball,
                                   dp.semantic_segmentation_metadata.human.head.eye.right.tear_duct,
                                   dp.semantic_segmentation_metadata.human.head.eye.right.eyelid])
    left_lip_colormap = np.array([dp.semantic_segmentation_metadata.human.head.mouth.lips.top.left,
                                  dp.semantic_segmentation_metadata.human.head.mouth.lips.bottom.left,
                                  dp.semantic_segmentation_metadata.human.head.mouth.teeth.bottom,
                                  dp.semantic_segmentation_metadata.human.head.mouth.teeth.top,
                                  dp.semantic_segmentation_metadata.human.head.mouth.gums.top,
                                  dp.semantic_segmentation_metadata.human.head.mouth.tongue,
                                  dp.semantic_segmentation_metadata.human.head.mouth.interior])
    right_lip_colormap = np.array([dp.semantic_segmentation_metadata.human.head.mouth.lips.top.right,
                                   dp.semantic_segmentation_metadata.human.head.mouth.lips.bottom.right,
                                   dp.semantic_segmentation_metadata.human.head.mouth.teeth.bottom,
                                   dp.semantic_segmentation_metadata.human.head.mouth.teeth.top,
                                   dp.semantic_segmentation_metadata.human.head.mouth.gums.bottom,
                                   dp.semantic_segmentation_metadata.human.head.mouth.tongue,
                                   dp.semantic_segmentation_metadata.human.head.mouth.interior])
    skin_color_map = np.array([dp.semantic_segmentation_metadata.human.head.skin.left,
                               dp.semantic_segmentation_metadata.human.head.skin.right])

    parts = [nose_colormap,
             eyebrow_colormap,
             left_eye_colormap,
             right_eye_colormap,
             left_lip_colormap,
             right_lip_colormap,
             skin_color_map]

    for part in parts:
        for color in part:
            mask = (dp.semantic_segmentation == color)
            seg_mask[mask[..., 0]] += 1

    hull = ConvexHull(face_dense_kpts_2d)
    face_mask = np.zeros_like(image)

    for simplex in hull.simplices:
        cv2.line(face_mask, (face_dense_kpts_2d[simplex, 1][0], face_dense_kpts_2d[simplex, 0][0]),
                 (face_dense_kpts_2d[simplex, 1][1], face_dense_kpts_2d[simplex, 0][1]),
                 color=(255, 255, 255),
                 thickness=1)

    face_mask = cv2.cvtColor(face_mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(image=face_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image=face_mask, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=cv2.FILLED,
                     lineType=cv2.LINE_AA)

    human_segment = cv2.bitwise_and(face_mask, face_mask, mask=seg_mask)
    perc_vis_face = (np.sum(human_segment)/np.sum(face_mask))*100

    return perc_vis_face


if __name__ == '__main__':

    print(f"{get_perc_vis_face(dp):0.1f}% of the face is visible")
