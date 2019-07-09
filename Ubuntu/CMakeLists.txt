# 按惯例，cmake的版本
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
# 项目名称
PROJECT(Convert)
# cmake寻找cuda，这个要现在系统里面装好cuda，设置好cuda的环境参数啥的
FIND_PACKAGE(CUDA REQUIRED)
# C++和CUDA的编译参数，可选。
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;-std=c++11;)
# 头文件路径，按需
INCLUDE_DIRECTORIES(
    .
    /usr/local/cuda-10.1/samples/common/inc
)
# 库文件路径，按需
LINK_DIRECTORIES(
	/usr/lib
   	/usr/local/lib
)
# 主要就是这个，教cmake去找nvcc来编译这些东西
CUDA_ADD_EXECUTABLE(Convert
    convertToP010.cu
    Source.cpp
)
# 链接外部库，按需
#TARGET_LINK_LIBRARIES(ad-census
#    某个库的名字)

