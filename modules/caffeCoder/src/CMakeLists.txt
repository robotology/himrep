# Copyright: (C) 2014 iCub Facility - Istituto Italiano di Tecnologia
# Authors: Giulia Pasquale
# CopyPolicy: Released under the terms of the GNU GPL v2.0.
cmake_minimum_required(VERSION 3.5)

set(PROJECTNAME caffeCoder)
project(${PROJECTNAME})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenCV REQUIRED)

include_directories(${Caffe_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR})

#if(NOT Caffe_CPU_ONLY)
  #no need to use the following instruction because this is automatically done by CUDA_ADD_EXECUTABLE
  #include_directories(${CUDA_INCLUDE_DIRS})
#endif()

get_target_property(CAFFE_SONAME caffe IMPORTED_SONAME_RELEASE)

MESSAGE( STATUS "CAFFE_SONAME:         " ${CAFFE_SONAME} )

if("${CAFFE_SONAME}" MATCHES "([0-9]+).([0-9]+).([0-9]+)")

    set(CAFFE_MAJOR "${CMAKE_MATCH_1}")
    set(CAFFE_MINOR "${CMAKE_MATCH_2}")
    set(CAFFE_PATCH "${CMAKE_MATCH_3}")

    MESSAGE( STATUS "CAFFE_MAJOR:         " ${CMAKE_MATCH_1} )
    MESSAGE( STATUS "CAFFE_MINOR:         " ${CMAKE_MATCH_2} )
    MESSAGE( STATUS "CAFFE_PATCH:         " ${CMAKE_MATCH_3} )

    set(CAFFE_VERSION_STRING "${CAFFE_MAJOR}.${CAFFE_MINOR}.${CAFFE_PATCH}")

else()

    set(CAFFE_MAJOR 0)
    set(CAFFE_MINOR 0)
    set(CAFFE_PATCH 0)

    set(CAFFE_VERSION_STRING unknown)

endif()

math(EXPR CAFFE_VERSION "${CAFFE_MAJOR}*10000 + ${CAFFE_MINOR}*100 + ${CAFFE_PATCH}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../caffe-version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/caffe-version.h
               @ONLY)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

set(folder_source main.cpp)
set(folder_header CaffeFeatExtractor.hpp)
source_group("Source Files" FILES ${folder_source})
source_group("Header Files" FILES ${folder_header})

if (NOT Caffe_CPU_ONLY)

    add_definitions(-DHAS_CUDA)

    find_package(CUDA REQUIRED)

    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)

endif()

if(NOT Caffe_CPU_ONLY)
    cuda_add_executable(${PROJECTNAME} ${folder_source} ${folder_header})
else()
    add_executable(${PROJECTNAME} ${folder_source} ${folder_header})
endif()

target_link_libraries(${PROJECTNAME} ${YARP_LIBRARIES} ${OpenCV_LIBRARIES} ${Caffe_LIBRARIES})

if(NOT Caffe_CPU_ONLY)
    target_link_libraries(${PROJECTNAME} ${CUDA_LIBRARIES})
endif()

install(TARGETS ${PROJECTNAME} DESTINATION bin)

yarp_install(FILES ../${PROJECTNAME}.xml DESTINATION ${ICUBCONTRIB_MODULES_INSTALL_DIR})
