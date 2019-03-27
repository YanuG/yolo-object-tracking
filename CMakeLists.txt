cmake_minimum_required(VERSION 2.8.3)
project(yolo_object_tracking)

## Compile as C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  sensor_msgs
  std_msgs  
  message_generation
  roslib
)

## Find CUDA 
find_package(CUDA REQUIRED)

## Find OpenCV
find_package(OpenCV REQUIRED)

# Find GNOME 
find_package(PkgConfig REQUIRED)
pkg_search_module(GLIB REQUIRED glib-2.0)

## Find TensorRT Directory and Libraries  
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN})
find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)

## Find CUDNN Directory and Libraries 
set(CUDNN_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA cuDNN")
find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES cuda/include include)
find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)
find_package_handle_standard_args(
    CUDNN DEFAULT_MSG CUDNN_INCLUDE_DIR CUDNN_LIBRARY)

## get all messages in msg folder:
file(GLOB messages
        "msg/*.msg"
        )
## strip path and add:
foreach(message ${messages})
    get_filename_component(m ${message} NAME)
    add_message_files(
            FILES
            ${m}
    )
endforeach(message ${messages})

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${CUDNN_INCLUDE_DIR}
  ${GLIB_INCLUDE_DIRS}
)

catkin_package( 
   INCLUDE_DIRS include
   LIBRARIES ${yolo_object_tracking}
   CATKIN_DEPENDS cv_bridge image_transport roscpp sensor_msgs std_msgs message_runtime
  )

## Gather all files in src directory
file(GLOB SOURCES "src/detector/*.cpp" )

add_executable(yolo_object_tracking ${SOURCES})

add_definitions(${GLIB_CFLAGS_OTHER})

## Generate library from .cu file
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62)
SET(LIB_TYPE STATIC) 
CUDA_ADD_LIBRARY(CUDA_LIBS ${LIB_TYPE} src/detector/kernels.cu )

## Link all libraries to code 
target_link_libraries(yolo_object_tracking  PRIVATE PRIVATE "stdc++fs" 
  ${GLIB_LIBRARIES}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${TENSORRT_LIBRARY}
  ${CUDNN_LIBRARY}
  CUDA_LIBS)
