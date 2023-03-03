cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_BUILD_TYPE "Debug")

project(nvdsvideotemplate-demo LANGUAGES CXX CUDA)

#message(STATUS "CMAKE_HOST_SYSTEM_PROCESSOR: ${CMAKE_HOST_SYSTEM_PROCESSOR}")

set(CMAKE_CXX_STANDARD 11)

# CUDA
set(CUDA_LIB "/usr/local/cuda/include/lib")
set(CUDA_INC "/usr/local/cuda/include")
find_package(CUDA REQUIRED)

# cudnn
set(CUDNN_INC "/usr/include/")

# TensorRT
set(TRT_INC "/usr/include/x86_64-linux-gnu/")
set(TRT_LIB "/usr/lib/x86_64-linux-gnu/")
set(TRT_OSS_LIB "/opt/tensorrt/lib/")

# Gstreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_BASE REQUIRED gstreamer-base-1.0)
pkg_check_modules(GSTREAMER_VIDEO REQUIRED gstreamer-video-1.0)

# DeepStream
set(DeepStream_DIR "/opt/nvidia/deepstream/deepstream/")
set(DeepStream_SRC_DIR ${DeepStream_DIR}/sources)
set(DeepStream_INC "${DeepStream_SRC_DIR}/includes")
set(DeepStream_NVDSVIDEOTEMPLATE_INC "${DeepStream_SRC_DIR}/gst-plugins/gst-nvdsvideotemplate/includes")
set(DeepStream_LIB "${DeepStream_DIR}/lib")
if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
set(Tegra_LIB "/usr/lib/aarch64-linux-gnu/tegra")
endif()

# try find deepstream version

set(DeepStream_VERSION_STRING "")

file(STRINGS ${DeepStream_DIR}/version DeepStream_VERSION_STRING REGEX "^Version:[ \t\r\n]*[0-9]+\\.[0-9]+\\.[0-9]+[ \t\r\n]*")

if (NOT DeepStream_VERSION_STRING)
file(STRINGS ${DeepStream_DIR}/version DeepStream_VERSION_STRING REGEX "^Version:[ \t\r\n]*[0-9]+\\.[0-9]+[ \t\r\n]*")
set(DeepStream_VERSION_STRING "${DeepStream_VERSION_STRING}.0")
#message(STATUS "string: ${DeepStream_VERSION_STRING}")
endif()

if (NOT DeepStream_VERSION_STRING)
message(WARNING "can note get DeepStream version from file ${DeepStream_DIR}/version")
set(DeepStream_VERSION_STRING "Version: 6.0.0")
endif()

#message(STATUS "DeepStream_VERSION_STRING: ${DeepStream_VERSION_STRING}")
string(REGEX REPLACE ".*([0-9]+)\\.[0-9]+\\.[0-9]+" "\\1" DeepStream_VERSION_MAJOR ${DeepStream_VERSION_STRING})
string(REGEX REPLACE ".*[0-9]+\\.([0-9]+)\\.[0-9]+" "\\1" DeepStream_VERSION_MINOR ${DeepStream_VERSION_STRING})
string(REGEX REPLACE ".*[0-9]+\\.[0-9]+\\.([0-9]+)" "\\1" DeepStream_VERSION_PATCH ${DeepStream_VERSION_STRING})

#message(STATUS "DeepStream_VERSION: ${DeepStream_VERSION_MAJOR}" "${DeepStream_VERSION_MINOR}" "${DeepStream_VERSION_PATCH}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDS_VERSION_MAJOR=${DeepStream_VERSION_MAJOR} -DDS_VERSION_MINOR=${DeepStream_VERSION_MINOR} -DDS_VERSION_PATCH=${DeepStream_VERSION_PATCH}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDS_VERSION_MAJOR=${DeepStream_VERSION_MAJOR} -DDS_VERSION_MINOR=${DeepStream_VERSION_MINOR} -DDS_VERSION_PATCH=${DeepStream_VERSION_PATCH}")

# opencv
find_package(OpenCV REQUIRED)

# libtorch
#set(CMAKE_PREFIX_PATH "/home/agx/miniforge-pypy3/envs/H/lib/python3.8/site-packages/torch/share/cmake")
#find_package(Torch REQUIRED)
#set(CMAKE_CXX_FLAGS "${CAMKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPLATFORM_TEGRA=1")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPLATFORM_TEGRA=1")
endif()

include_directories(${TRT_INC}  ${CUDA_INC} ${CUDNN_INC} ${OpenCV_INCLUDE_DIRS} ${GSTREAMER_INCLUDE_DIRS} ${DeepStream_INC} ${DeepStream_NVDSVIDEOTEMPLATE_INC})
link_directories(${TRT_LIB}  ${CUDA_LIB} ${TRT_OSS_LIB})
if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
link_directories(${Tegra_LIB})
endif()

# customer lib
set(CUSTOMER_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nvdsvideotemplate)
set(CUSTOMER_LIB_SRCs   ${CUSTOMER_LIB_DIR}/customlib_impl/customlib_impl.cpp
                        ${CUSTOMER_LIB_DIR}/common/algorithm/cuda/nv12_to_rgb_kernel.cu
                        ${CUSTOMER_LIB_DIR}/trt/logger.cpp)
set(CUSTOMER_LIB_INC ${CUSTOMER_LIB_DIR}/includes ${CUSTOMER_LIB_DIR}/trt ${CUSTOMER_LIB_DIR}/common/algorithm/cuda)
set(DeepStream_LIBRARIES nvdsgst_helper nvdsgst_meta nvds_meta nvbufsurface nvbufsurftransform nvdsbufferpool)
add_library(customlib_videoimpl SHARED ${CUSTOMER_LIB_SRCs})
if(${CMAKE_VERSION} VERSION_LESS 3.13)
if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
set(CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} "-L${DeepStream_LIB} -Wl,-rpath,${DeepStream_LIB}")
else()
set(CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} "-Wl,--no-undefined -Wl,-rpath,$(DeepStream_LIB)")
endif()
link_directories(${DeepStream_LIB})
else()
#if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
#target_link_options(customlib_videoimpl PUBLIC -Wl,-rpath,$(DeepStream_LIB))
#else()
target_link_options(customlib_videoimpl PUBLIC -Wl,--no-undefined -Wl,-rpath,$(DeepStream_LIB))
#endif()
target_link_directories(customlib_videoimpl PUBLIC ${DeepStream_LIB})
endif()
target_include_directories(customlib_videoimpl PUBLIC ${CUSTOMER_LIB_INC} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(customlib_videoimpl   ${CUDA_LIBRARIES}
                                            cuda
                                            nvinfer
                                            nvonnxparser
                                            #nvcaffeparser
                                            ${CUDA_nppc_LIBRARY}
                                            ${CUDA_nppial_LIBRARY}
                                            ${DeepStream_LIBRARIES}
                                            ${GSTREAMER_LIBRARIES}
                                            ${GSTREAMER_BASE_LIBRARIES}
                                            ${GSTREAMER_VIDEO_LIBRARIES}
                                            ${OpenCV_LIBS}
                                            ${TORCH_LIBRARIES})

#add_executable(run_demo test.cpp demo.cpp)


#target_link_libraries(run_demo "nvinfer" "nvinfer_plugin" "nvonnxparser" )
#target_link_libraries(run_demo ${CUDA_LIBRARIES} ${OpenCV_LIBS})
#target_link_libraries(run_demo ${TORCH_LIBRARIES})

#set_property(TARGET run_demo PROPERTY CXX_STANDARD 14)

# python samples
set(PYTHON_SAMPLES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

FILE(GLOB PYTHON_COMMON_SOURCES "${PYTHON_SAMPLES_DIR}/common/*.py")
#FOREACH(python_source ${PYTHON_COMMON_SOURCES})
#    FILE(COPY ${python_source} DESTINATION ./common)
#ENDFOREACH(python_source)

add_custom_target(python_common
    ALL
    DEPENDS ${PYTHON_COMMON_SOURCES})
add_custom_command(
    TARGET python_common
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${PYTHON_SAMPLES_DIR}/common
    ${CMAKE_CURRENT_BINARY_DIR}/common
)

FILE(GLOB PYTHON_SOURCES "${PYTHON_SAMPLES_DIR}/*.py")

add_custom_target(python_samples
    ALL
    DEPENDS ${PYTHON_SOURCES} ${PYTHON_COMMON_SOURCES})

add_custom_command(
    TARGET python_samples
    COMMAND cp ${PYTHON_SOURCES} ${CMAKE_CURRENT_BINARY_DIR}
)
