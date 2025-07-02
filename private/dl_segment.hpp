#ifndef DL_SEGMENT_HPP
#define DL_SEGMENT_HPP

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <NvInfer.h>

#include "../include/dl_base.hpp"

namespace fs = std::filesystem;


class SegResult final : public BaseResult {
public:
    SegResult(cv::Mat boxes, const std::vector<cv::Mat>& masks);

    bool extractSegResult(cv::Mat& boxes, std::vector<cv::Mat>& masks) const override;

private:
    cv::Mat boxes;
    std::vector<cv::Mat> masks;
};


// 部署模型基类
class SegDeployModel : public BaseDeployModel {
public:
    explicit SegDeployModel(const CfgType& cfg);

    ~SegDeployModel() override;

    std::unique_ptr<BaseResult> operator()(const cv::Mat& im0) override;

    [[nodiscard]] std::tuple<cv::Mat, float, cv::Point2f> preprocess(const cv::Mat& img) const;

    virtual std::vector<cv::Mat> inference(const cv::Mat& img) = 0;

    [[nodiscard]] std::tuple<cv::Mat, std::vector<cv::Mat>> postprocess(const std::vector<cv::Mat>& preds,
                                                                        const cv::Mat& img) const;

    void process_box(const cv::Mat& res) const;

    static std::vector<cv::Mat> process_mask(const cv::Mat& protos, const cv::Mat& masks_coef, const cv::Mat& boxes,
                                             const cv::Size& shape);

    static std::vector<cv::Mat> scale_mask(const std::vector<cv::Mat>& masks, const cv::Size& im0_shape);

    static std::vector<cv::Mat> crop_mask(const std::vector<cv::Mat>& masks, const cv::Mat& boxes);

    static std::vector<cv::Mat> masks2segments(const std::vector<cv::Mat>& masks);

protected:
    std::string deploy_name;
    std::vector<std::string> classes;
    int model_w;
    int model_h;
    int nm = 32;
    float conf;
    float iou;
    int img_w{};
    int img_h{};
    float scale_r{};
    float pad_w{};
    float pad_h{};

private:
    static AlgorithmType algorithm_type;
    static bool register_status;
};


// TensorRT 模型类
class SegTensorRtModel final : public SegDeployModel {
public:
    explicit SegTensorRtModel(const CfgType& cfg);

    ~SegTensorRtModel() override;

    std::vector<cv::Mat> inference(const cv::Mat& img) override;

private:
    struct Binding;

    class Logger;
    std::unique_ptr<Logger> logger;

    std::string input_name;
    std::vector<std::string> output_names;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    std::unordered_map<std::string, Binding*> bindings;
    cudaStream_t stream{};
};

#endif //DL_SEGMENT_HPP
