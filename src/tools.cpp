#include <nlohmann/json.hpp>
#include <fstream>

#include "../private/tools.hpp"

using json = nlohmann::json;


// 将 json 值转换为 std::any
std::any Tools::json_to_any(const json& j) {
    if (j.is_null()) {
        return nullptr;
    }
    if (j.is_boolean()) {
        return j.get<bool>();
    }
    if (j.is_number_integer()) {
        return j.get<int>();
    }
    if (j.is_number_float()) {
        return j.get<float>();
    }
    if (j.is_string()) {
        return j.get<std::string>();
    }
    if (j.is_array()) {
        // 处理数组：递归转换每个元素
        std::vector<std::any> arr;
        for (const auto& ele : j) {
            arr.push_back(json_to_any(ele));
        }
        return arr;
    }
    throw std::runtime_error("Unsupported JSON type");
}

CfgType Tools::parse_json_config(const std::string& config_path) {
    std::ifstream file(config_path);
    auto json_cfg = json::parse(file);
    CfgType config;
    // 读取每个变量
    for (auto& var : json_cfg) {
        // 解析 name 和 value
        std::string name;
        std::any value;
        for (auto& item : var.items()) {
            if ("name" == item.key()) {
                name = item.value();
            } else if ("value" == item.key()) {
                value = json_to_any(item.value());
            }
        }
        config[name] = value;
    }
    return config;
}

bool Tools::check_gpu() {
    return true;
}

void Tools::adaptive_show(const cv::Mat& img, const bool resize_window) {
    cv::namedWindow("eu_box", cv::WINDOW_NORMAL);
    if (resize_window) {
        cv::resizeWindow("eu_box", 1280, 960);
    }
    cv::imshow("eu_box", img);
    cv::waitKey();
}

void Tools::save_image(const std::string& root, const std::string& name, const cv::Mat& image, const bool flag) {
    if (flag) {
        fs::create_directories(root);
        cv::imwrite(root + "/" + name, image);
    }
}

cv::Mat Tools::transpose(const cv::Mat& input, const std::vector<int>& newOrder) {
    /*
     将 a*b*c 的矩阵任意交换维度
     */
    // 将通道也单独作为一维
    cv::Mat src = input.clone();
    if (input.channels() == 3) {
        auto h = input.rows, w = input.cols;
        src = input.reshape(1, {h, w, 3});
    }
    CV_Assert(src.dims == 3 && src.isContinuous());
    auto dims = src.size;
    CV_Assert(dims.dims() == 3 && newOrder.size() == 3);

    // (a, b, c) -> newOrder
    cv::Mat dst({dims[newOrder[0]], dims[newOrder[1]], dims[newOrder[2]]}, src.type());
    for (int i = 0; i < dims[0]; ++i) {
        for (int j = 0; j < dims[1]; ++j) {
            for (int k = 0; k < dims[2]; ++k) {
                std::vector indices = {i, j, k};
                memcpy(dst.ptr(indices[newOrder[0]], indices[newOrder[1]], indices[newOrder[2]]),
                       src.ptr(i, j, k), src.elemSize());
            }
        }
    }
    return dst;
}

cv::Mat Tools::vecNHW2HWNmat(const std::vector<cv::Mat>& input) {
    /*
     将 (N, H, W) 的 vector<cv::Mat> 转化为 (H, W, N) 的 cv::Mat
     */
    int N = input.size();
    auto H = input[0].rows, W = input[0].cols;
    cv::Mat dst({H, W, N}, input[0].type());
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                *dst.ptr<uchar>(i, j, k) = *input[k].ptr(i, j);
            }
        }
    }
    return dst;
}

std::vector<cv::Mat> Tools::matHWN2NHWvec(const cv::Mat& input) {
    /*
     将 (H, W, N) 的 cv::Mat 转化为 (N, H, W) 的 vector<cv::Mat>
     */
    const auto H = input.size[0], W = input.size[1], N = input.size[2];
    std::vector<cv::Mat> dst(N);
    for (int k = 0; k < N; ++k) {
        dst[k] = cv::Mat(H, W, input.type());
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                *dst[k].ptr(i, j) = *input.ptr<uchar>(i, j, k);
            }
        }
    }
    return dst;
}
