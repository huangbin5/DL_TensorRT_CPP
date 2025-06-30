#include "../include/dl_base.hpp"

bool BaseResult::extractSegResult(cv::Mat& boxes, std::vector<cv::Mat>& masks) const {
    return false;
}

bool BaseResult::extractDetResult(cv::Mat& boxes) const {
    return false;
}

bool BaseResult::extractClsResult(std::vector<float>& confs) const {
    return false;
}


std::unique_ptr<BaseDeployModel> BaseDeployModel::create(const AlgorithmType type, const CfgType& cfg) {
    auto registry = getRegistry();
    if (const auto it = registry.find(type); it != registry.end()) {
        return it->second(cfg);
    }
    return nullptr;
}

void BaseDeployModel::registerType(const AlgorithmType type, const CreateFunc& func) {
    getRegistry()[type] = func;
}

std::unordered_map<AlgorithmType, BaseDeployModel::CreateFunc>& BaseDeployModel::getRegistry() {
    static std::unordered_map<AlgorithmType, CreateFunc> registry;
    return registry;
}
