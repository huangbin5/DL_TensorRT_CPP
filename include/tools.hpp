#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>

using namespace std;
namespace fs = filesystem;


class Tools {
public:
    static bool check_gpu();

    static void adaptive_show(const cv::Mat& img, bool resize_window = true);

    static void save_image(const string& root, const string& name, const cv::Mat& image, bool flag = true);

    // 以下方法暂时用不上，但已经实现了就先放着了
    static cv::Mat transpose(const cv::Mat& input, const vector<int>& newOrder);

    static cv::Mat vecNHW2HWNmat(const vector<cv::Mat>& input);

    static vector<cv::Mat> matHWN2NHWvec(const cv::Mat& input);
};

#endif // TOOLS_HPP
