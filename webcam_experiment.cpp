
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#define MARKER_SIZE         0.03f

using namespace cv;
using namespace cv::aruco;
using namespace std;

int main(void) {
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    Ptr<Dictionary> tag_dict = getPredefinedDictionary(DICT_6X6_1000);
    vector<vector<Point2f>> rejected_marker_corners;
    vector<vector<Point2f>> marker_corners;
    vector<int> marker_ids;
    Ptr<DetectorParameters> det_params = DetectorParameters::create();
    vector<Vec3d> rvecs, tvecs;
    Mat camera_params, dist_params;

    namedWindow("frame",1);
    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera

        detectMarkers(frame, tag_dict, marker_corners, marker_ids, det_params, rejected_marker_corners);
        if (marker_corners.size() != 0) {
            drawDetectedMarkers(frame, marker_corners, marker_ids);
            estimatePoseSingleMarkers(marker_corners, MARKER_SIZE, camera_params, dist_params, rvecs, tvecs);
            for (size_t i = 0; i < marker_ids.size(); ++i)
                drawAxis(frame, camera_params, dist_params, rvecs[i], tvecs[i], 0.5);
        } else
            drawDetectedMarkers(frame, rejected_marker_corners, noArray(), Scalar(0, 0, 255));

        imshow("frame", frame);
        if ((waitKey(1) & 0xff) == 'q')
            break;
    }

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
