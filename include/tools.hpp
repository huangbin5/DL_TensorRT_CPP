#ifndef TOOLS_HPP
#define TOOLS_HPP
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;


class Tools {
public:
    static cv::Mat transpose(const cv::Mat& input, const vector<int>& newOrder);

    static cv::Mat vecNHW2HWNmat(const vector<cv::Mat>& input);

    static vector<cv::Mat> matHWN2NHWvec(const cv::Mat& input);

    static void adaptive_show(const cv::Mat& img, bool resize_window = true);
};

#endif // TOOLS_HPP
