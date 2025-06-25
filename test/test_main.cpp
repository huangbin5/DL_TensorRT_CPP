#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <unordered_map>

#include "test_segment.hpp"

using namespace std;


int main() {
    const CfgType config = {
        {"model_path", string("/home/oyefish/data/eu_box_exp/exp03_0403_lvl/train6/weights/best.engine")},
        {"classes", vector<string>{"eu_box"}},
        {"model_w", 1280},
        {"model_h", 1280},
        {"nm", 32},
        {"conf", 0.25f},
        {"iou", 0.7f},
    };
    SegTest model(config,
                  "/home/oyefish/data/eu_box/0324_test",
                  true, true, true, false);
    const auto start = chrono::high_resolution_clock::now();
    model.infer_batch();
    // model.infer_single("Image_2025-03-24_14_52_34.png");
    const auto end = chrono::high_resolution_clock::now();
    cout << "一共耗时 " << chrono::duration_cast<chrono::seconds>(end - start).count() << 's' << endl;
    return 0;
}
