#ifndef DL_SEGMENT_HPP
#define DL_SEGMENT_HPP

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <NvInfer.h>

#include "dl_base.hpp"

using namespace std;


class SegResult final : public BaseResult {
public:
    SegResult(const cv::Mat& boxes, const vector<cv::Mat>& masks);

    bool extractSegResult(cv::Mat& boxes, vector<cv::Mat>& masks) const override;

private:
    cv::Mat boxes;
    vector<cv::Mat> masks;
};


// 部署模型基类
class SegDeployModel : public BaseDeployModel {
public:
    static bool register_status;

    explicit SegDeployModel(const CfgType& cfg);

    ~SegDeployModel() override;

    unique_ptr<BaseResult> operator()(const cv::Mat& im0) override;

    [[nodiscard]] tuple<cv::Mat, float, cv::Point2f> preprocess(const cv::Mat& img) const;

    virtual vector<cv::Mat> inference(const cv::Mat& img) = 0;

    [[nodiscard]] tuple<cv::Mat, vector<cv::Mat>> postprocess(const vector<cv::Mat>& preds, const cv::Mat& img) const;

    void process_box(const cv::Mat& res) const;

    static vector<cv::Mat> process_mask(const cv::Mat& protos, const cv::Mat& masks_coef, const cv::Mat& boxes,
                                        const cv::Size& shape);

    static vector<cv::Mat> scale_mask(const vector<cv::Mat>& masks, const cv::Size& im0_shape);

    static vector<cv::Mat> crop_mask(const vector<cv::Mat>& masks, const cv::Mat& boxes);

    static vector<cv::Mat> masks2segments(const vector<cv::Mat>& masks);

protected:
    string deploy_name;
    vector<string> classes;
    int model_w;
    int model_h;
    int nm;
    float conf;
    float iou;
    int img_w{};
    int img_h{};
    float scale_r{};
    float pad_w{};
    float pad_h{};
};


// TensorRT 模型类
class SegTensorRtModel final : public SegDeployModel {
public:
    explicit SegTensorRtModel(const CfgType& cfg);

    ~SegTensorRtModel() override;

    vector<cv::Mat> inference(const cv::Mat& img) override;

private:
    struct Binding;

    class Logger;
    unique_ptr<Logger> logger;

    string input_name;
    vector<string> output_names;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    unordered_map<string, Binding*> bindings;
    cudaStream_t stream{};
};

#endif //DL_SEGMENT_HPP
