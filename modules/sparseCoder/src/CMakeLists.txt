# Copyright: (C) 2011 Department of Robotics Brain and Cognitive Sciences - Istituto Italiano di Tecnologia
# Authors: Carlo Ciliberto
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

cmake_minimum_required(VERSION 2.6)
set(PROJECTNAME sparseCoder)
project(${PROJECTNAME})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(SIFTGPU)

if(SIFTGPU_FOUND) 
   find_package(OpenCV)
   if(OpenCV_VERSION_MAJOR GREATER 2)
      message(STATUS "detected Version(OpenCV) > 2.x.x => ${PROJECTNAME} will not be built)
   else()
      include_directories(${YARP_INCLUDE_DIRS})
      include_directories(${OpenCV_INCLUDE_DIRS})
      
      set(folder_source main.cpp SiftGPU_Extractor.cpp DictionaryLearning.cpp)
      set(folder_header SiftGPU_Extractor.h DictionaryLearning.h)
      source_group("Source Files" FILES ${folder_source})
      source_group("Header Files" FILES ${folder_header})
      include_directories(${PROJECT_SOURCE_DIR} ${SIFTGPU_INCLUDE_DIRS})
      add_executable(${PROJECTNAME} ${folder_source} ${folder_header})
      target_link_libraries(${PROJECTNAME} ${OpenCV_LIBRARIES} ${YARP_LIBRARIES} ${SIFTGPU_LIBRARIES} ${CMAKE_DL_LIBS})
      install(TARGETS ${PROJECTNAME} DESTINATION bin)
   endif()
else()
   message(STATUS "SIFTGPU not found => ${PROJECTNAME} will not be built)
endif()

