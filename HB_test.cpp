#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

int main() {
    cv::Mat src(2, 3,CV_8U);
    cout << src.channels() << endl;
    *src.ptr(0) = 1;
    cout << src << endl;
    cout << "HB" << endl;
    return 0;
}
