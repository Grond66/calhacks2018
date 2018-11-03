
from cv2 import *
from cv2.aruco import *

cam = VideoCapture(0)
tag_dict = getPredefinedDictionary(DICT_4X4_1000)

namedWindow("frame", 1)

while True:
    ret, frame = cam.read()

    marker_corners, marker_ids, rejected_marker_corners = detectMarkers(frame, tag_dict)
    drawDetectedMarkers(frame, marker_corners, marker_ids)

    imshow("frame", frame)

    if waitKey(1) & 0xFF == ord('q'):
        break
