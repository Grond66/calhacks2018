
from cv2 import *
from cv2.aruco import *
import pickle

cam = VideoCapture(0)
tag_dict = getPredefinedDictionary(DICT_4X4_50)

camera_params = dist_params = None
with open("calibration.pikl") as f:
    camera_params, dist_params = pickle.load(f)

namedWindow("frame", 1)

while True:
    ret, frame = cam.read()

    marker_corners, marker_ids, rejected_marker_corners = detectMarkers(frame, tag_dict)
    drawDetectedMarkers(frame, marker_corners, marker_ids)

    if len(marker_corners) > 0:
        rvecs, tvecs = estimatePoseSingleMarkers(marker_corners, 0.03, camera_params, dist_params)

        i = 0
        while i < len(tvecs):
            drawAxis(frame, camera_params, dist_params, rvecs[i], tvecs[i], 0.07)
            i = i + 1

    imshow("frame", frame)

    if waitKey(1) & 0xff == ord('q'):
        break
