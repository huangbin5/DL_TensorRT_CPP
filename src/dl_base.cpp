#include "../include/dl_base.hpp"


unique_ptr<BaseDeployModel> BaseDeployModel::create(const AlgorithmType type, const CfgType& cfg) {
    auto registry = getRegistry();
    if (const auto it = registry.find(type); it != registry.end()) {
        return it->second(cfg);
    }
    return nullptr;
}

void BaseDeployModel::registerType(const AlgorithmType type, const CreateFunc& func) {
    // 注意：使用 move 提高效率
    getRegistry()[type] = move(func);
}

unordered_map<AlgorithmType, BaseDeployModel::CreateFunc>& BaseDeployModel::getRegistry() {
    static unordered_map<AlgorithmType, CreateFunc> registry;
    return registry;
}
