# Copyright: (C) 2014 iCub Facility - Istituto Italiano di Tecnologia
# Authors: Giulia Pasquale
# CopyPolicy: Released under the terms of the GNU GPL v2.0.
cmake_minimum_required(VERSION 2.8.8)

set(PROJECTNAME caffeCoder)
project(${PROJECTNAME})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenCV REQUIRED)

# CUDA packagee
find_package(CUDA REQUIRED)
# CUDA compilation flags
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)

include_directories(${YARP_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Caffe_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR})

# no need to use the following instruction because this is automatically done by CUDA_ADD_EXECUTABLE
#if(NOT CPU_ONLY)
#  include_directories(${CUDA_INCLUDE_DIRS})
#endif()

add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

set(folder_source main.cpp)
set(folder_header CaffeFeatExtractor.hpp)
source_group("Source Files" FILES ${folder_source})
source_group("Header Files" FILES ${folder_header})

cuda_add_executable(${PROJECTNAME} ${folder_source} ${folder_header})

target_link_libraries(${PROJECTNAME} ${YARP_LIBRARIES} ${OpenCV_LIBRARIES} ${Caffe_LIBRARIES})
target_link_libraries(${PROJECTNAME} ${CUDA_LIBRARIES})

install(TARGETS ${PROJECTNAME} DESTINATION bin)
