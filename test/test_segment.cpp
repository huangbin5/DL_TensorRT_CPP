#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <any>
#include <filesystem>
#include <numeric>

#include "../include/tools.hpp"
#include "../include/dl_segment.hpp"
#include "../include/test_segment.hpp"


SegTest::SegTest(const CfgType& cfg, const string& exp_root, const bool save_mask, const bool save_box,
                 const bool save_conf, const bool show_result)
    : _model(BaseDeployModel::create(DL_SEGMENT, cfg)),
      _classes(any_cast<vector<string>>(cfg.at("classes"))),
      _exp_root(exp_root),
      _save_mask(save_mask),
      _save_box(save_box),
      _save_conf(save_conf),
      _show_result(show_result) {
    if (!exp_root.empty()) {
        save_root = exp_root + "/arun";
    }
    _colors = {{0, 255, 0}, {0, 0, 255}, {0, 255, 255}, {255, 0, 0}, {255, 0, 255}};
}

/**
 * 暂时只支持处理一张图片
 * Args:
 *     bgr_image: (1536, 2048, 3)
 * Return:
 *     masks_per_label: list[ndarray]. 每个类别的 mask 存储为一个 ndarray (m, h, w), 其中 m 为 mask 个数
 */
tuple<vector<vector<cv::Mat>>, vector<vector<cv::Rect2f>>, vector<vector<float>>> SegTest::get_mask(
    const cv::Mat& bgr_image) const {
    // 只测试 TensorRT
    /*
    boxes: (m, 6) (y1, x1, y2, x2, conf, cls)
    masks: (m, h, w)
     */
    // 注意：这里必须分两个步骤得到 results，如果是一步的话，uni_ptr 是临时变量会被删除
    const auto uni_ptr = (*_model)(bgr_image);
    const auto results = static_cast<SegResult*>(uni_ptr.get());
    const auto boxes = results->boxes;
    const auto masks = results->masks;

    vector<vector<cv::Mat>> masks_per_label(_classes.size());
    vector<vector<cv::Rect2f>> boxes_per_label(_classes.size());
    vector<vector<float>> confs_per_label(_classes.size());
    if (boxes.rows > 0) {
        // 将每个标签的物体分开
        for (size_t cls = 0; cls < _classes.size(); ++cls) {
            vector<cv::Mat> class_masks;
            vector<cv::Rect2f> class_boxes;
            vector<float> class_confs;
            for (int i = 0; i < boxes.rows; ++i) {
                if (boxes.at<float>(i, 5) == static_cast<float>(cls)) {
                    cv::Mat mask = masks[i].clone();
                    mask.convertTo(mask, CV_8U, 255);
                    // 将 mask 缩放至原图大小
                    cv::Mat new_mask;
                    cv::resize(mask, new_mask, bgr_image.size(), 0, 0, cv::INTER_NEAREST);
                    class_masks.push_back(new_mask);
                    // 因为是 Rect 类型，需要转化为宽和高
                    class_boxes.emplace_back(boxes.at<float>(i, 0), boxes.at<float>(i, 1),
                                             boxes.at<float>(i, 2) - boxes.at<float>(i, 0),
                                             boxes.at<float>(i, 3) - boxes.at<float>(i, 1));
                    class_confs.push_back(boxes.at<float>(i, 4));
                }
            }
            masks_per_label[cls] = class_masks;
            boxes_per_label[cls] = class_boxes;
            confs_per_label[cls] = class_confs;
        }
    } else {
        cerr << "未识别物料" << endl;
        for (size_t i = 0; i < _classes.size(); ++i) {
            masks_per_label[i] = {};
            boxes_per_label[i] = {};
            confs_per_label[i] = {};
        }
    }

    return {masks_per_label, boxes_per_label, confs_per_label};
}

bool SegTest::infer_single(const string& img_name) {
    if (_exp_root.empty()) {
        throw runtime_error("推理数据的目录不能为空");
    }
    cout << "filename: " << img_name << " ";
    cv::Mat bgr_image = cv::imread(_exp_root + "/" + img_name);
    int bgr_h = bgr_image.rows, bgr_w = bgr_image.cols;
    auto [masks_per_label, boxes_per_label,
        confs_per_label] = get_mask(bgr_image);

    cv::Mat result_image = bgr_image.clone();
    cv::Mat mask_binary = cv::Mat::zeros(bgr_h, bgr_w, CV_8U);
    cv::Mat box_binary = cv::Mat::zeros(bgr_h, bgr_w, CV_8U);
    // EU 箱只有一个类别，未来再扩充到多个类别
    vector<cv::Mat> masks = masks_per_label[0];
    vector<cv::Rect2f> boxes = boxes_per_label[0];
    vector<float> confs = confs_per_label[0];

    if (!masks.empty()) {
        // 需要整体的掩膜、边界框、置信度
        for (int k = 0; k < masks.size(); ++k) {
            int border = 3;
            const cv::Mat& mask = masks[k];
            const cv::Rect2f box = boxes[k];
            // 将 mask 画在原图上
            for (int i = 0; i < result_image.rows; ++i) {
                auto* row_img = result_image.ptr<cv::Vec3b>(i);
                const auto* row_mask = mask.ptr<uchar>(i);
                for (int j = 0; j < result_image.cols; ++j) {
                    if (row_mask[j] == 255) {
                        cv::Vec3b& pixel = row_img[j];
                        pixel[0] = cv::saturate_cast<uchar>(pixel[0] * 0.7 + _colors[k][0] * 0.3);
                        pixel[1] = cv::saturate_cast<uchar>(pixel[1] * 0.7 + _colors[k][1] * 0.3);
                        pixel[2] = cv::saturate_cast<uchar>(pixel[2] * 0.7 + _colors[k][2] * 0.3);
                    }
                }
            }
            mask_binary.setTo(255, mask == 255);
            // box 二值图
            box_binary(box).setTo(255);
            auto inner_box = cv::Rect2f(box.x + border, box.y + border,
                                        box.width - 2 * border, box.height - 2 * border);
            box_binary(inner_box).setTo(0);
            if (_save_conf) {
                cv::putText(result_image, cv::format("%.2f", confs[k]),
                            cv::Point(box.x + box.width / 2 - 30, box.y + box.height / 2 + 15),
                            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 255), 2);
            }
        }
        Tools::save_image(save_root + "/result", fs::path(img_name).stem().string() + "_result.png", result_image);
        Tools::save_image(save_root, fs::path(img_name).stem().string() + "_masks.png", mask_binary, _save_mask);
        Tools::save_image(save_root, fs::path(img_name).stem().string() + "_boxes.png", box_binary, _save_box);
    } else {
        Tools::save_image(save_root + "/result", fs::path(img_name).stem().string() + "_result.png", bgr_image);
        Tools::save_image(save_root + "/failed", img_name, bgr_image);
    }
    if (_show_result) {
        Tools::adaptive_show(result_image);
    }
    return !masks.empty();
}

void SegTest::infer_batch() {
    vector<string> rgb_files;
    for (const auto& entry : fs::directory_iterator(_exp_root)) {
        if (const string file = entry.path().filename().string(); file.ends_with(".png")) {
            rgb_files.push_back(file);
        }
    }
    sort(rgb_files.begin(), rgb_files.end());

    const int all_cnt = rgb_files.size();
    int failed = 0;
    for (auto i = 0; i < all_cnt; ++i) {
        cout << "\n【Image " << i + 1 << '/' << all_cnt << "】\t";
        failed += !infer_single(rgb_files[i]);
    }
    cout << "\nsegment saved to " << save_root << endl;
    cout << "failed " << failed << '/' << all_cnt << '=' << cv::format("%.2f", 100.0 * failed / all_cnt) << "%\n";
}
