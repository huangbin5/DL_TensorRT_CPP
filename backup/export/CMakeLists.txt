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

# ------------------- 输出目录配置 -------------------
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ------------------- 收集头文件和源代码 -------------------
set(PUBLIC_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)  # 公共头文件（仅暴露dl_base.hpp）
set(PRIVATE_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/private)  # 私有头文件（内部使用）
file(GLOB_RECURSE SOURCE_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)

# ------------------- 创建动态库 -------------------
add_library(dl_base SHARED ${SOURCE_FILES})
set_target_properties(dl_base PROPERTIES PUBLIC_HEADER "${PUBLIC_INCLUDE_DIR}/include/dl_base.hpp")


# ------------------- 包含目录处理（替代原include_directories） -------------------
target_include_directories(dl_base
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>  # 构建时的路径
        $<INSTALL_INTERFACE:include>                    # 安装后的路径
        PRIVATE
        ${PRIVATE_INCLUDE_DIR}         # 私有头文件
        ${CUDA_DIR}/include
        ${OnnxRuntime_DIR}/include
        ${TensorRT_DIR}/include
)
# ------------------- 链接库处理（替代原link_directories） -------------------
if (WIN32)
    target_link_libraries(dl_base
            ${OpenCV_LIBS}
            ${CUDA_DIR}/lib/x64/cudart.lib        # 原link_directories路径+库名
            ${OnnxRuntime_DIR}/lib/onnxruntime.lib
            ${TensorRT_DIR}/lib/nvinfer.lib
            ${TensorRT_DIR}/lib/nvinfer_plugin.lib
    )
else ()
    target_link_libraries(dl_base
            ${OpenCV_LIBS}
            ${CUDA_DIR}/lib64/libcudart.so        # 原link_directories路径+库名
            ${OnnxRuntime_DIR}/lib/libonnxruntime.so
            ${TensorRT_DIR}/lib/libnvinfer.so
            ${TensorRT_DIR}/lib/libnvinfer_plugin.so
    )
endif ()
# ------------------- 安装与导出配置 -------------------
install(TARGETS dl_base
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include
)