#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <nlohmann/json.hpp>

#include "../include/dl_base.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;


class Tools {
public:
    static std::any json_to_any(const json& j);

    static CfgType parse_json_config(const std::string& config_path);

    template <typename T>
    static std::vector<T> any_to_vector(const std::any& value) {
        const auto& vec_any = std::any_cast<std::vector<std::any>>(value);
        std::vector<std::string> result;
        result.reserve(vec_any.size());

        for (const auto& ele : vec_any) {
            result.push_back(std::any_cast<std::string>(ele));
        }
        return result;
    }

    static bool check_gpu();

    static void adaptive_show(const cv::Mat& img, bool resize_window = true);

    static void save_image(const std::string& root, const std::string& name, const cv::Mat& image, bool flag = true);

    // 以下方法暂时用不上，但已经实现了就先放着了
    static cv::Mat transpose(const cv::Mat& input, const std::vector<int>& newOrder);

    static cv::Mat vecNHW2HWNmat(const std::vector<cv::Mat>& input);

    static std::vector<cv::Mat> matHWN2NHWvec(const cv::Mat& input);
};

#endif // TOOLS_HPP
