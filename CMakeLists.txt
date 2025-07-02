cmake_minimum_required(VERSION 3.16.3)
set(CMAKE_CXX_STANDARD 20)
project(DL_TensorRT_CPP)

if (WIN32)
    # 设置 OpenCV 路径
    set(OpenCV_DIR "D:/software/opencv/build")
    # 设置 CUDA 路径
    set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
    # 设置 ONNX 路径
    set(OnnxRuntime_DIR "D:/software/onnxruntime-win-x64-gpu-1.22.0")
    # 设置 TensorRT 路径
    set(TensorRT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-8.6.1.6")
else ()
    set(CUDA_DIR "/usr/local/cuda")
    set(OnnxRuntime_DIR "/home/oyefish/application/onnxruntime")
    set(TensorRT_DIR "/home/oyefish/application/TensorRT")
endif ()
# 查找OpenCV库
find_package(OpenCV REQUIRED)

# 包含头文件目录
include_directories(
        ${CUDA_DIR}/include
        ${OnnxRuntime_DIR}/include
        ${TensorRT_DIR}/include
)
# 添加库文件目录
link_directories(
        ${OnnxRuntime_DIR}/lib
        ${TensorRT_DIR}/lib
)
if (WIN32)
    link_directories(
            ${CUDA_DIR}/lib/x64
    )
else ()
    link_directories(
            ${CUDA_DIR}/lib64
    )
endif ()


# 需要将所有使用到的 cpp 文件添加进去
add_executable(deep_learning
        src/dl_base.cpp
        src/dl_segment.cpp
        src/tools.cpp
        test/test_main.cpp
        test/test_segment.cpp
)
# 链接第三方库
target_link_libraries(deep_learning
        ${OpenCV_LIBS}
        cudart         # CUDA运行时库
        onnxruntime    # ONNX Runtime库
        nvinfer        # TensorRT核心库
        nvinfer_plugin
)


# 所有 cpp 文件不依赖其它自定义文件时，可以这样批量使用
#file(GLOB_RECURSE file_paths *.cpp)
#foreach (file_path ${file_paths})
#    get_filename_component(file_name ${file_path} NAME_WE)
#    add_executable(${file_name} ${file_path})
#    message(${file_name} ${file_path})
#
#    # 链接OpenCV库
#    target_link_libraries(${file_name}
#            ${OpenCV_LIBS}
#            cudart         # CUDA运行时库
#            onnxruntime    # ONNX Runtime库
#            nvinfer        # TensorRT核心库
#            nvinfer_plugin
#    )
#endforeach ()