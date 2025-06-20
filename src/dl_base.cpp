#include "../include/dl_base.hpp"


void BaseDeployModel::registerType(const AlgorithmType type, const CreateFunc func) {
    registry[type] = func;
}

BaseDeployModel* BaseDeployModel::create(const AlgorithmType type) {
    if (const auto it = registry.find(type); it != registry.end()) {
        return it->second();
    }
    return nullptr;
}
