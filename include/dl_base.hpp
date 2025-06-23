#ifndef DL_BASE_HPP
#define DL_BASE_HPP

#include <opencv2/opencv.hpp>
#include <any>

using namespace std;


using CfgType = unordered_map<string, any>;
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

    // 工厂方法
    static unique_ptr<BaseDeployModel> create(AlgorithmType type, const CfgType& cfg);

protected:
    // 注意：存储函数指针而不是基类指针的原因是每次调用都创建一个新的对象
    // 注意：使用 function 比函数指针更好
    using CreateFunc = function<unique_ptr<BaseDeployModel>(CfgType)>;

    // 注册派生类构建函数
    static void registerType(AlgorithmType type, const CreateFunc& func);

private:
    // 获取注册表。注意：使用 get 方法获取，确保在 registerType 的时候 registry 一定存在
    static unordered_map<AlgorithmType, CreateFunc>& getRegistry();
};

#endif //DL_BASE_HPP
