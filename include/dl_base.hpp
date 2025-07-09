#ifndef DL_BASE_HPP
#define DL_BASE_HPP

#include <opencv2/opencv.hpp>
#include <any>


// 通过 EXPORT 标记的符号（类、函数、变量）会被添加到 DLL 的导出列表中
// 允许外部程序通过 DLL 调用这些功能
#if defined(_MSC_VER) // 比 ifdef 支持更复杂的表达式
    #define EXPORT __declspec(dllexport) // Windows
#else
    #define EXPORT __attribute__((visibility("default"))) // Linux
#endif


using CfgType = std::unordered_map<std::string, std::any>;

enum class AlgorithmType {
    DL_CLASSIFY = 0,
    DL_DETECT = 1,
    DL_SEGMENT = 2,
};


// 推理结果类
class EXPORT BaseResult {
public:
    virtual ~BaseResult() = default;

    virtual bool extractSegResult(cv::Mat& boxes, std::vector<cv::Mat>& masks) const;

    virtual bool extractDetResult(cv::Mat& boxes) const;

    virtual bool extractClsResult(std::vector<float>& confs) const;
};


// 推理模型基类
class EXPORT BaseDeployModel {
public:
    virtual ~BaseDeployModel() = default;

    virtual std::unique_ptr<BaseResult> operator()(const cv::Mat& im0) = 0;

    // 工厂方法
    static std::unique_ptr<BaseDeployModel> create(AlgorithmType type, const CfgType& cfg);

protected:
    // 注意：存储函数指针而不是基类指针的原因是每次调用都创建一个新的对象
    // 注意：使用 function 比函数指针更好
    using CreateFunc = std::function<std::unique_ptr<BaseDeployModel>(CfgType)>;

    // 注册派生类构建函数
    static void registerType(AlgorithmType type, const CreateFunc& func);

private:
    // 获取注册表。注意：使用 get 方法获取，确保在 registerType 的时候 registry 一定存在
    static std::unordered_map<AlgorithmType, CreateFunc>& getRegistry();
};

#endif //DL_BASE_HPP
