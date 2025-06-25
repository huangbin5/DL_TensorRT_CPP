#ifndef TEST_SEGMENT_HPP
#define TEST_SEGMENT_HPP

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "dl_base.hpp"

using namespace std;


class SegTest {
public:
    explicit SegTest(const unordered_map<string, any>& cfg, const string& exp_root = "", bool save_mask = true, bool save_box = false,
                     bool save_conf = true, bool show_result = false);

    tuple<vector<vector<cv::Mat>>, vector<vector<cv::Rect2f>>, vector<vector<float>>> get_mask(
        const cv::Mat& bgr_image) const;

    bool infer_single(const string& img_name);

    void infer_batch();

private:
    unique_ptr<BaseDeployModel> _model;
    vector<string> _classes;
    string _exp_root;
    string save_root;
    vector<cv::Scalar> _colors;
    bool _save_mask;
    bool _save_box;
    bool _save_conf;
    bool _show_result;
};

#endif // TEST_SEGMENT_HPP
