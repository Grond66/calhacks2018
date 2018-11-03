
from cv2 import *
from cv2.aruco import *
import pickle

number_of_calibration_pictures = 10

cam = VideoCapture(0)
tag_dict = getPredefinedDictionary(DICT_4X4_50)
charuco_board = CharucoBoard_create(5, 7, 0.04, 0.03, tag_dict)

def get_charuco_grid(iteration, total):
    window_name = "take calibration picture " + str(iteration) + "/" + str(total)
    namedWindow(window_name, 1)

    while True:
        ret, frame = cam.read()

        imshow(window_name, frame);

        if waitKey(1) & 0xFF == ord('y'):
            break

    marker_corners, marker_ids, rejected_marker_corners = detectMarkers(frame, tag_dict)
    num_corners, charuco_corners, charuco_ids = interpolateCornersCharuco(marker_corners, marker_ids, frame, charuco_board)

    drawDetectedMarkers(frame, marker_corners, marker_ids)
    drawDetectedCornersCharuco(frame, charuco_corners)

    imshow(window_name, frame)

    waitKey(0)

    destroyWindow(window_name)

    return charuco_corners, charuco_ids

def get_image_size():
    ret, frame = cam.read()
    (height, width, _) = frame.shape
    return height, width

detected_boards_list = []
detected_boards_ids_list = []
for i in range(number_of_calibration_pictures):
    board, ids = get_charuco_grid(i, number_of_calibration_pictures)
    detected_boards_list.append(board)
    detected_boards_ids_list.append(ids)

reprojection_error, camera_params, dist_params, _, _ = calibrateCameraCharuco(detected_boards_list, detected_boards_ids_list, charuco_board, get_image_size(), None, None)

with open("calibration.pikl", "wb") as f:
    pickle.dump((camera_params, dist_params), f)
