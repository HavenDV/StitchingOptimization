#ifndef StitchingHeader
#define StitchingHeader

#include <opencv2/opencv.hpp>

cv::Mat startStitching();

void addCameraParameters(cv::Mat &image, int HFOV, cv::Mat rotationMatrix);
void startRefiningCameraParameters();

int main();

#endif
