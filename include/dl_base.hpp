#ifndef DL_BASE_HPP
#define DL_BASE_HPP

#include <opencv2/opencv.hpp>

using namespace std;


enum AlgorithmType {
    DL_CLASSIFY = 0,
    DL_DETECT = 1,
    DL_SEGMENT = 2,
};

class BaseResult {
};


class BaseDeployModel {
public:
    virtual ~BaseDeployModel() = default;

    virtual unique_ptr<BaseResult> operator()(const cv::Mat& im0) = 0;

    // 函数指针类型
    using CreateFunc = BaseDeployModel*(*)();
    // 注册派生类构建函数
    static void registerType(AlgorithmType type, CreateFunc func);
    // 工厂方法
    static BaseDeployModel* create(AlgorithmType type);

private:
    static std::unordered_map<AlgorithmType, CreateFunc> registry;
};

#endif //DL_BASE_HPP
