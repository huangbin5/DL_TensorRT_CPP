#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>
#include <NvInfer.h>
#include <any>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <chrono>
#include <numeric>

#include "../include/tools.hpp"
#include "../include/dl_segment.hpp"

namespace fs = filesystem;


SegResult::SegResult(const cv::Mat& boxes, const vector<cv::Mat>& masks) : boxes(boxes), masks(masks) {
}

bool SegResult::extractSegResult(cv::Mat& boxes, vector<cv::Mat>& masks) const {
    boxes = this->boxes;
    masks = this->masks;
    return true;
}

bool SegDeployModel::register_status = [] {
    // todo 判断是否有 GPU
    if (Tools::check_gpu()) {
        registerType(AlgorithmType::DL_SEGMENT, [](CfgType cfg) { return make_unique<SegTensorRtModel>(cfg); });
    }
    return true;
}();

SegDeployModel::SegDeployModel(const CfgType& cfg)
    : classes(any_cast<vector<string>>(cfg.at("classes"))),
      model_w(any_cast<int>(cfg.at("model_w"))),
      model_h(any_cast<int>(cfg.at("model_h"))),
      nm(any_cast<int>(cfg.at("nm"))),
      conf(any_cast<float>(cfg.at("conf"))),
      iou(any_cast<float>(cfg.at("iou"))) {
}

SegDeployModel::~SegDeployModel() = default;

unique_ptr<BaseResult> SegDeployModel::operator()(const cv::Mat& im0) {
    img_h = im0.rows, img_w = im0.cols;
    auto [img, r, pad] = preprocess(im0);
    scale_r = r, pad_w = pad.x, pad_h = pad.y;
    const auto start_time = chrono::high_resolution_clock::now();
    // input: (1, 3, 1280, 1280)
    // output: ((37, 33600), (32, 320, 320))
    const auto preds = inference(img);
    const chrono::duration<double> elapsed = chrono::high_resolution_clock::now() - start_time;
    cout << deploy_name << " 推理时间：" << elapsed.count() << "s\n";
    auto [boxes, masks] = postprocess(preds, im0);
    return make_unique<SegResult>(boxes, masks);
}

[[nodiscard]] tuple<cv::Mat, float, cv::Point2f> SegDeployModel::preprocess(const cv::Mat& img) const {
    const cv::Size shape(img.cols, img.rows);
    const cv::Size new_shape(model_w, model_h);
    float r = min(static_cast<float>(new_shape.height) / static_cast<float>(shape.height),
                  static_cast<float>(new_shape.width) / static_cast<float>(shape.width));
    const cv::Size unpad(
        static_cast<int>(round(static_cast<float>(shape.width) * r)),
        static_cast<int>(round(static_cast<float>(shape.height) * r))
    );
    cv::Mat resized_img;
    if (shape != unpad) {
        cv::resize(img, resized_img, unpad, 0, 0, cv::INTER_LINEAR);
    } else {
        resized_img = img.clone();
    }
    const float pad_w = static_cast<float>(new_shape.width - unpad.width) / 2.0f;
    const float pad_h = static_cast<float>(new_shape.height - unpad.height) / 2.0f;
    const int top = static_cast<int>(round(pad_h - 0.1));
    const int bottom = static_cast<int>(round(pad_h + 0.1));
    const int left = static_cast<int>(round(pad_w - 0.1));
    const int right = static_cast<int>(round(pad_w + 0.1));
    cv::Mat padded_img;
    cv::copyMakeBorder(resized_img, padded_img, top, bottom, left, right, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));

    // Transforms: HWC to CHW -> BGR to RGB -> contiguous -> div(255) -> add axis
    padded_img.convertTo(padded_img, CV_32F, 1.0 / 255.0);
    cv::Mat img_process;
    cv::dnn::blobFromImage(padded_img, img_process, 1.0, cv::Size(), cv::Scalar(), true, false);
    return {img_process, r, cv::Point2f(pad_w, pad_h)};
}

/**
 * (res, protos) = preds
 * res.shape = (37, 33600)
 * protos.shape = (32, 320, 320)
 */
[[nodiscard]] tuple<cv::Mat, vector<cv::Mat>> SegDeployModel::postprocess(
    const vector<cv::Mat>& preds, const cv::Mat& img) const {
    cv::Mat res = preds[0].clone();
    cv::Mat protos = preds[1].clone();
    // (37, 33600) -> (33600, 37)  4边界框(xc, yc, w, h) + 1类别 + 32掩膜系数
    cv::transpose(res, res);

    // (20, 37) 按置信度过滤
    cv::Mat scores = res.colRange(4, res.cols - nm);
    cv::Mat max_scores;
    cv::reduce(scores, max_scores, 1, cv::REDUCE_MAX);
    cv::Mat valid_indices;
    cv::findNonZero(max_scores > conf, valid_indices); // findNonZero 得到的类型是 Point 坐标
    cv::Mat valid_res(valid_indices.rows, res.cols, res.type());
    for (int i = 0; i < valid_indices.rows; ++i) {
        int index = valid_indices.at<cv::Point>(i).y; // 获取行号
        res.row(index).copyTo(valid_res.row(i));
    }

    // (20, 38)  4边界框 + 1置信度 + 1最大下标0 + 32掩膜系数
    cv::Mat detail_res;
    for (int i = 0; i < valid_res.rows; ++i) {
        cv::Mat row = valid_res.row(i).clone();
        cv::Mat class_scores = row.colRange(4, row.cols - nm);
        double max_val;
        cv::Point max_loc;
        cv::minMaxLoc(class_scores, nullptr, &max_val, nullptr, &max_loc);

        cv::Mat new_row = cv::Mat::zeros(1, 6 + nm, row.type());
        row.colRange(0, 4).copyTo(new_row.colRange(0, 4));
        new_row.at<float>(0, 4) = static_cast<float>(max_val);
        new_row.at<float>(0, 5) = static_cast<float>(max_loc.x); // 注意 Point (或者说 OpenCV) 中的 x 表示列，y 表示行
        row.colRange(row.cols - nm, row.cols).copyTo(new_row.colRange(6, 6 + nm));
        detail_res.push_back(new_row);
    }

    // (2, 38) 非极大值抑制过滤
    vector<int> indices;
    vector<cv::Rect2d> boxes;
    vector<float> confidences;
    for (int i = 0; i < detail_res.rows; ++i) {
        cv::Vec4f box = detail_res.row(i).colRange(0, 4);
        boxes.emplace_back(box[0], box[1], box[2], box[3]); // 直接传递构造参数，效率更高
        confidences.push_back(detail_res.at<float>(i, 4));
    }
    cv::dnn::NMSBoxes(boxes, confidences, conf, iou, indices);
    if (indices.empty()) {
        // return {cv::Mat(), cv::Mat(), {}};
        return {cv::Mat(), cv::Mat()};
    }
    cv::Mat final_res;
    for (int index : indices) {
        final_res.push_back(detail_res.row(index));
    }

    process_box(final_res);
    vector<cv::Mat> masks = process_mask(
        protos, final_res.colRange(final_res.cols - nm, final_res.cols),
        final_res.colRange(0, 4), img.size()
    );
    // 暂时用不上
    // vector<cv::Mat> segments = masks2segments(masks);
    // return {final_res.colRange(0, 6), masks, segments};
    return {final_res.colRange(0, 6), masks};
}

void SegDeployModel::process_box(const cv::Mat& res) const {
    res.col(0) -= res.col(2) / 2;
    res.col(1) -= res.col(3) / 2;
    res.col(2) += res.col(0);
    res.col(3) += res.col(1);
    res.col(0) -= pad_w;
    res.col(1) -= pad_h;
    res.col(2) -= pad_w;
    res.col(3) -= pad_h;
    res.col(0) /= scale_r;
    res.col(1) /= scale_r;
    res.col(2) /= scale_r;
    res.col(3) /= scale_r;
    res.col(0) = cv::max(res.col(0), 0);
    res.col(1) = cv::max(res.col(1), 0);
    res.col(2) = cv::min(res.col(2), img_w);
    res.col(3) = cv::min(res.col(3), img_h);
}

/**
 * @param protos (32, 320, 320)
 * @param masks_coef (2, 32)
 * @param boxes (2, 4)
 * @param shape (1536, 2048, 3)
 */
vector<cv::Mat> SegDeployModel::process_mask(const cv::Mat& protos, const cv::Mat& masks_coef, const cv::Mat& boxes,
                                             const cv::Size& shape) {
    const int c = protos.size[0], mh = protos.size[1], mw = protos.size[2]; // 多维数组不能用 rows 和 cols，会直接返回 -1
    // (320, 320, 2) 根据模板生成掩膜
    cv::Mat masks;
    // (2, 32) * (32, H * W) -> (2, H * W)
    cv::gemm(masks_coef, protos.reshape(0, c), 1, cv::Mat(), 0, masks);
    // (2, H * W) -> (2, H, W)
    masks = masks.reshape(0, {masks_coef.rows, mh, mw});

    // 2 * (H, W)
    vector<cv::Mat> vec_masks(masks.size[0]);
    // 注意：以下初始化写法是错误的，Mat 的拷贝构造函数是浅拷贝，vec_masks 中所有元素都会指向同一个 Mat
    // vector vec_masks(masks.size[0], cv::Mat(masks.size[1], masks.size[2], masks.type()));
    for (int k = 0; k < vec_masks.size(); ++k) {
        vec_masks[k] = cv::Mat(masks.size[1], masks.size[2], masks.type());
        memcpy(vec_masks[k].ptr(), masks.ptr(k), vec_masks[k].total() * CV_ELEM_SIZE(masks.type()));
    }
    vec_masks = scale_mask(vec_masks, shape);
    vec_masks = crop_mask(vec_masks, boxes);
    for (auto& mask : vec_masks) {
        cv::threshold(mask, mask, 0.5, 1, cv::THRESH_BINARY);
    }
    return vec_masks;
}

/**
 * 先将 mask 填充至原图比例，再缩放至原图大小
 * @param masks 2 * (320, 320)
 * @param im0_shape (1536, 2048, 3)
 */
vector<cv::Mat> SegDeployModel::scale_mask(const vector<cv::Mat>& masks, const cv::Size& im0_shape) {
    const auto height = masks[0].size[0], width = masks[0].size[1];
    const float r = min(height * 1.0 / im0_shape.height, width * 1.0 / im0_shape.width);
    const float pad_h = (height - im0_shape.height * r) / 2;
    const float pad_w = (width - im0_shape.width * r) / 2;
    const int top = static_cast<int>(round(pad_h - 0.1));
    const int left = static_cast<int>(round(pad_w - 0.1));
    const int bottom = static_cast<int>(round(height - pad_h + 0.1));
    const int right = static_cast<int>(round(width - pad_w + 0.1));

    vector<cv::Mat> dst(masks.size());
    for (int k = 0; k < masks.size(); ++k) {
        dst[k] = cv::Mat(height, width, masks[0].type());
        const cv::Mat cropped_masks = masks[k](cv::Rect(left, top, right - left, bottom - top));
        cv::resize(cropped_masks, dst[k], im0_shape, 0, 0, cv::INTER_LINEAR);
    }
    return dst;
}

/**
 * 裁剪 bbox 外部的区域，确保 bbox 完全包裹 mask
 * @param masks (2, 1536, 2048)
 * @param boxes (2, 4)
 */
vector<cv::Mat> SegDeployModel::crop_mask(const vector<cv::Mat>& masks, const cv::Mat& boxes) {
    // cout << boxes << endl;
    const int n = masks.size();
    const int h = masks[0].size[0], w = masks[0].size[1];
    vector<cv::Mat> dst(n);
    for (int k = 0; k < n; ++k) {
        dst[k] = cv::Mat::zeros(h, w, masks[0].type());
        const auto row = boxes.ptr<float>(k);
        cv::Rect roi(row[0], row[1], row[2] - row[0] + 1, row[3] - row[1] + 1);
        masks[k](roi).copyTo(dst[k](roi));
    }
    return dst;
}

/**
 * 将 mask 掩膜转化为 contour 边界点
 * @param masks (2, 1536, 2048)
 * @return
 */
vector<cv::Mat> SegDeployModel::masks2segments(const vector<cv::Mat>& masks) {
    vector<cv::Mat> segments_list;
    for (const auto& k : masks) {
        cv::Mat mask = k.clone();
        mask.convertTo(mask, CV_8U);
        vector<vector<cv::Point>> contours;
        // RETR_EXTERNAL 只保留外边界
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        if (!contours.empty()) {
            // 只保存最大的边界（主要物体而非噪声），x 为轮廓，x.shape[0] 为轮廓点数
            size_t max_index = 0, max_size = 0;
            for (size_t i = 0; i < contours.size(); ++i) {
                if (contours[i].size() > max_size) {
                    max_size = contours[i].size();
                    max_index = i;
                }
            }
            // (n, 1, 2) -> (n, 2)
            const auto& max_contour = contours[max_index];
            cv::Mat seg(max_contour.size(), 2, CV_32S);
            for (size_t i = 0; i < max_contour.size(); ++i) {
                seg.at<float>(i, 0) = max_contour[i].y;
                seg.at<float>(i, 1) = max_contour[i].x;
            }
            segments_list.push_back(seg);
        } else {
            segments_list.emplace_back(0, 2, CV_32S);
        }
    }
    return segments_list;
}


struct SegTensorRtModel::Binding {
    void* host;
    void* device;
    size_t size;
};

class SegTensorRtModel::Logger final : public nvinfer1::ILogger {
public:
    void log(const Severity severity, const char* msg) noexcept override {
        // 忽略 INFO 以下级别的日志
        if (severity <= Severity::kINFO) {
            std::cout << "[TRT LOG] " << msg << std::endl;
        }
    }
};

SegTensorRtModel::SegTensorRtModel(const CfgType& cfg) : SegDeployModel(cfg), logger(make_unique<Logger>()) {
    deploy_name = "tensorrt";
    input_name = "images";
    output_names = {"output0", "output1"};

    // 读取序列化文件
    auto tensorrt_path = any_cast<string>(cfg.at("model_path"));
    ifstream file(tensorrt_path, ios::binary);
    if (!file) {
        throw runtime_error("Failed to open TensorRT engine file");
    }

    /*
     YOLO 导出的模型需要去除 Meta 信息
     */
    // 读取 Meta 数据的长度
    uint32_t meta_len;
    file.read(reinterpret_cast<char*>(&meta_len), sizeof(meta_len));
    // 跳过 Meta 数据
    file.seekg(meta_len, ios::cur);
    size_t engine_data_start = file.tellg();

    file.seekg(0, ios::end);
    size_t size = file.tellg();
    size -= engine_data_start;
    // 回到 engine 实际数据处
    file.seekg(engine_data_start, ios::beg);
    vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    // 反序列化 engine。Logger -> Runtime -> Engine -> Context
    auto runtime = nvinfer1::createInferRuntime(*logger);
    engine = runtime->deserializeCudaEngine(engine_data.data(), size);
    context = engine->createExecutionContext();

    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        auto name = engine->getIOTensorName(i);
        auto [nbDims, d] = engine->getTensorShape(name);
        // nvinfer1::DataType dtype = engine->getTensorDataType(name); // todo 先不使用 dtype，直接用 float
        size_t p = accumulate(d, d + nbDims, 1, [](auto a, auto b) { return a * b; });
        // 分配锁页内存，避免被操作系统换出，提升DMA传输效率
        void* host_mem;
        cudaHostAlloc(&host_mem, p * sizeof(float), cudaHostAllocDefault);
        // 预分配显存块，减少运行时内存碎片
        void* device_mem;
        cudaMalloc(&device_mem, p * sizeof(float));
        context->setTensorAddress(name, device_mem);
        bindings[name] = new Binding{host_mem, device_mem, p};
    }
    cudaStreamCreate(&stream);
}

SegTensorRtModel::~SegTensorRtModel() {
    for (auto& [name, addr] : bindings) {
        cudaFreeHost(addr->host);
        cudaFree(addr->device);
    }
    cudaStreamDestroy(stream);
    delete context;
    delete engine;
}

/**
 * @param img (1, 3, 1280, 1280)
 * @return ((37, 33600), (32, 320, 320))
 */
vector<cv::Mat> SegTensorRtModel::inference(const cv::Mat& img) {
    const auto input = bindings[input_name];
    const auto output0 = bindings[output_names[0]];
    const auto output1 = bindings[output_names[1]];

    memcpy(input->host, img.ptr<float>(), img.total() * sizeof(float));
    cudaMemcpyAsync(input->device, input->host, img.total() * sizeof(float), cudaMemcpyHostToDevice, stream);
    context->enqueueV3(stream);
    cudaMemcpyAsync(output0->host, output0->device, output0->size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(output1->host, output1->device, output1->size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cv::Mat output0_mat(37, 33600, CV_32F, output0->host);
    cv::Mat output1_mat({32, 320, 320}, CV_32F, output1->host);
    return {output0_mat, output1_mat};
}
