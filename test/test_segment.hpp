#ifndef TEST_SEGMENT_HPP
#define TEST_SEGMENT_HPP

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "../include/dl_base.hpp"


class SegTest {
public:
    explicit SegTest(const std::unordered_map<std::string, std::any>& cfg, const std::string& exp_root = "",
                     bool save_mask = true, bool save_box = false, bool save_conf = true, bool show_result = false);

    [[nodiscard]] std::tuple<std::vector<std::vector<cv::Mat>>, std::vector<std::vector<cv::Rect2f>>, std::vector<
                                 std::vector<float>>> get_mask(const cv::Mat& bgr_image) const;

    bool infer_single(const std::string& img_name);

    void infer_batch();

private:
    std::unique_ptr<BaseDeployModel> _model;
    std::vector<std::string> _classes;
    std::string _exp_root;
    std::string save_root;
    std::vector<cv::Scalar> _colors;
    bool _save_mask;
    bool _save_box;
    bool _save_conf;
    bool _show_result;
};

#endif // TEST_SEGMENT_HPP
