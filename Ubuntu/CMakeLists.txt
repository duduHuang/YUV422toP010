CMAKE_MINIMUM_REQUIRED(VERSION 3.11)
PROJECT(duduconvert)
INCLUDE(CheckSymbolExists)

OPTION(BUILD_NVJPEG "Build with nvJPEG support" ON)

# Cmake path
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

# For NVJPEG
INCLUDE(cmake/Dependencies.cmake)

# CMAKE BUILD FLAGS
SET(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -Wall -g -DNDEBUG -D_DEBUG -Wconversion -std=c++11 -pthread -Wno-multichar")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -DNDEBUG -D_DEBUG -Wconversion -std=c++11 -pthread -Wno-multichar")

# For CUDA NVCC COMPILE FLAGS
INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CUDA_utils.cmake)

CUDA_find_supported_arch_values(CUDA_supported_archs ${CUDA_known_archs})
MESSAGE(STATUS "CUDA supported archs: ${CUDA_supported_archs}")

SET(CUDA_TARGET_ARCHS_SORTED ${CUDA_TARGET_ARCHS})
LIST(SORT CUDA_TARGET_ARCHS_SORTED)
CUDA_find_supported_arch_values(CUDA_targeted_archs ${CUDA_TARGET_ARCHS_SORTED})
MESSAGE(STATUS "CUDA targeted archs: ${CUDA_targeted_archs}")
IF (NOT CUDA_targeted_archs)
  MESSAGE(FATAL_ERROR "None of the provided CUDA architectures ({${CUDA_TARGET_ARCHS}}) is supported by nvcc, use one or more of: ${CUDA_supported_archs}")
ENDIF()

CUDA_get_gencode_args(CUDA_gencode_flags ${CUDA_targeted_archs})
MESSAGE(STATUS "Generated gencode flags: ${CUDA_gencode_flags}")

# Add ptx & bin flags for cuda
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_gencode_flags}")

FILE(GLOB CPU_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/writeFile.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/convertTool.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/DuDuConvertAPI.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/DuDuRGBConvertAPI.cpp"
)
FILE(GLOB GPU_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

FILE(GLOB LIBDPX_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/libdpx/*.cpp"
)

INCLUDE_DIRECTORIES(
	.
	/usr/local/cuda-10.1/samples/common/inc
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${CMAKE_CURRENT_SOURCE_DIR}/libdpx
)

SET(BUILD_SHARED_LIBS ON)

LINK_DIRECTORIES(
	/usr/lib
   	/usr/local/lib
)

CUDA_ADD_LIBRARY(duduconvert
    ${CPU_SRC}
    ${GPU_SRC}
	${LIBDPX_SRC}
)

# For debugging
MESSAGE(STATUS "CUDA_TOOLKIT_ROOT_DIR:${CUDA_TOOLKIT_ROOT_DIR}")
MESSAGE(STATUS "NVJPEG_LIBRARY:${NVJPEG_LIBRARY}")
MESSAGE(STATUS "CULIBOS_LIBRARY:${CULIBOS_LIBRARY}")

TARGET_LINK_LIBRARIES(duduconvert
	${NVJPEG_LIBRARY}
    ${CULIBOS_LIBRARY}
)

INCLUDE(GNUInstallDirs)

INSTALL(TARGETS duduconvert
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

INSTALL(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/DuDuConvertAPI.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/DuDuRGBConvertAPI.h
    DESTINATION include
    )

SET(prefix ${CMAKE_INSTALL_PREFIX})
SET(exec_prefix "\${prefix}")
SET(libdir ${CMAKE_INSTALL_LIBDIR})
SET(includedir "\${prefix}/include")
