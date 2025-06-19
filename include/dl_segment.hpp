#ifndef DL_SEGMENT_HPP
#define DL_SEGMENT_HPP

using namespace std;


// 部署模型基类
class DeployModel {
public:
    explicit DeployModel(const unordered_map<std::string, std::any>& cfg);

    virtual ~DeployModel();

    tuple<cv::Mat, vector<cv::Mat>> operator()(
        const cv::Mat& im0, float conf = 0.25, float iou = 0.7, int nm = 32);

    tuple<cv::Mat, float, cv::Point2f> preprocess(const cv::Mat& img) const;

    virtual vector<cv::Mat> inference(const cv::Mat& img) = 0;

    tuple<cv::Mat, vector<cv::Mat>> postprocess(
        const vector<cv::Mat>& preds, const cv::Mat& img, float conf, float iou, int nm = 32) const;

    void process_box(const cv::Mat& res) const;

    static vector<cv::Mat> process_mask(const cv::Mat& protos, const cv::Mat& masks_coef, const cv::Mat& bboxes,
                                        const cv::Size& shape);

    static vector<cv::Mat> scale_mask(const vector<cv::Mat>& masks, const cv::Size& im0_shape);

    static vector<cv::Mat> crop_mask(const vector<cv::Mat>& masks, const cv::Mat& boxes);

    static vector<cv::Mat> masks2segments(const vector<cv::Mat>& masks);

protected:
    string deploy_name;
    vector<string> classes;
    int model_w;
    int model_h;
    int img_w{};
    int img_h{};
    float scale_r{};
    float pad_w{};
    float pad_h{};
};


// TensorRT 模型类
class TensorRtModel final : public DeployModel {
public:
    explicit TensorRtModel(const unordered_map<string, any>& cfg);

    ~TensorRtModel() override;

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
