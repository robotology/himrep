# Copyright: (C) 2014 iCub Facility - Istituto Italiano di Tecnologia
# Authors: Giulia Pasquale
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

cmake_minimum_required(VERSION 2.8.8)

set(PROJECTNAME GIECoder)
project(${PROJECTNAME})

# setup GIE
#set(GIE_PATH "$ENV{ROBOT_CODE}/GIE" CACHE FILEPATH "Path to GPU Inference Engine")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")	# -std=gnu++11

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenCV REQUIRED)

# CUDA packagee
find_package(CUDA REQUIRED)

# CUDA compilation flags
set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS}; 
       -O3 -gencode arch=compute_53,code=sm_53
)

include_directories(${YARP_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

#include_directories(${GIE_PATH}/include)
#link_directories(${GIE_PATH}/lib)

include_directories(${PROJECT_SOURCE_DIR})

set(folder_source main.cpp GIEFeatExtractor.cpp )
set(folder_header GIEFeatExtractor.h cudaUtility.h)
source_group("Source Files" FILES ${folder_source})
source_group("Header Files" FILES ${folder_header})

cuda_add_executable(${PROJECTNAME} ${folder_source} ${folder_header})

target_link_libraries(${PROJECTNAME} ${YARP_LIBRARIES} ${OpenCV_LIBRARIES})
target_link_libraries(${PROJECTNAME} ${CUDA_LIBRARIES} nvcaffe_parser nvinfer)

install(TARGETS ${PROJECTNAME} DESTINATION bin)
yarp_install(FILES ../${PROJECTNAME}.xml DESTINATION ${ICUBCONTRIB_MODULES_INSTALL_DIR})
