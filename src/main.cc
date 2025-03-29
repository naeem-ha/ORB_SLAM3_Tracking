#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include "ORBextractor.h"

using namespace std;
using namespace cv;

int main() {
    Mat image = imread("/home/hamza-naeem/Documents/ORB_SLAM3_Tracking/data/1403636579813555456.png", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "Error: Could not load image!" << endl;
        return -1;
    }
    cout << "Image loaded: " << image.rows << "x" << image.cols << endl;

    int nFeatures = 1000;
    float scaleFactor = 1.2f;
    int nLevels = 8;
    int iniThFAST = 20;
    int minThFAST = 7;

    cout << "Constructing ORBextractor..." << endl;
    ORB_SLAM3::ORBextractor orbExtractor(nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST);
    cout << "ORBextractor initialized" << endl;

    vector<KeyPoint> keypoints;
    Mat descriptors = Mat::zeros(1000, 32, CV_8U);
    vector<int> vLappingArea;
    cout << "Calling ORBextractor..." << endl;
    orbExtractor(image, Mat(), keypoints, descriptors, vLappingArea);
    cout << "Features extracted" << endl;

    cout << "Extracted " << keypoints.size() << " ORB features." << endl;

    Mat outputImage;
    drawKeypoints(image, keypoints, outputImage, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imwrite("../data/output_frame1.jpg", outputImage);
    cout << "Image saved" << endl;

    return 0;
}