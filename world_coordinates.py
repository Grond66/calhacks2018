
from cv2 import *
from cv2.aruco import *
import pickle
import numpy as np

cam = VideoCapture(0)
tag_dict = getPredefinedDictionary(DICT_4X4_50)
charuco_board = CharucoBoard_create(5, 7, 0.04, 0.03, tag_dict)

camera_params = dist_params = None
with open("calibration.pikl") as f:
    camera_params, dist_params = pickle.load(f)

namedWindow("frame", 1)

def get_rt_matrix(rvec, tvec):
    rmat, _ = Rodrigues(rvec)
    tvec = tvec.transpose()
    body = np.concatenate((rmat, tvec), 1);
    bottom_row = np.zeros((1, 3))
    bottom_row = np.concatenate((bottom_row, np.ones((1, 1))), 1)
    return np.concatenate((body, bottom_row), 0)

while True:
    ret, frame = cam.read()

    marker_corners, marker_ids, rejected_marker_corners = detectMarkers(frame, tag_dict)

    if len(marker_corners) <= 0:
        frame = np.array(np.flip(frame, 1))
        imshow("frame", frame)
        if waitKey(1) & 0xff == ord('q'):
            break
        continue

    drawDetectedMarkers(frame, marker_corners, marker_ids)

    eraser_corners = None
    reference_corners = None
    for i in range(len(marker_ids)):
        if marker_ids[i] == 3:
            eraser_corners = marker_corners[i]
        elif marker_ids[i] == 1:
            reference_corners = marker_corners[i]

    tag_corners_list = []
    if eraser_corners is not None:
        tag_corners_list.append(eraser_corners)
    if reference_corners is not None:
        tag_corners_list.append(reference_corners)

    rvecs, tvecs = estimatePoseSingleMarkers(tag_corners_list, 0.03, camera_params, dist_params)

    if rvecs is None or tvecs is None:
        frame = np.array(np.flip(frame, 1))
        imshow("frame", frame)
        if waitKey(1) & 0xff == ord('q'):
            break
        continue

    for i in range(len(rvecs)):
        drawAxis(frame, camera_params, dist_params, rvecs[i], tvecs[i], 0.03)

    if eraser_corners is None or reference_corners is None:
        frame = np.array(np.flip(frame, 1))
        imshow("frame", frame)
        if waitKey(1) & 0xff == ord('q'):
            break
        continue

    eraser_rvec = rvecs[0]
    eraser_tvec = tvecs[0]
    reference_rvec = rvecs[1]
    reference_tvec = tvecs[1]

    eraser_x = eraser_tvec[0][0]
    eraser_y = eraser_tvec[0][1]
    eraser_z = eraser_tvec[0][2]

    print("({}, {}, {})".format(eraser_x, eraser_y, eraser_z))

    eraser_rt = get_rt_matrix(eraser_rvec, eraser_tvec)

    #eraser_rmat, _ = Rodrigues(eraser_rvec)
    #eraser_tmat = translation_matrix_from_translation_vector(eraser_tvec)
    #reference_rmat, _ = Rodrigues(reference_rvec)
    #reference_tmat = translation_matrix_from_translation_vector(reference_tvec)

    print(eraser_rt)

    drawAxis(frame, camera_params, dist_params, reference_rvec.transpose(), reference_tvec + np.array([0, 0.03, 0]), 0.03)

    frame = np.array(np.flip(frame, 1))
    imshow("frame", frame)
    if waitKey(1) & 0xff == ord('q'):
        break
