# Copyright: 2014 iCub Facility
# Author: Sean Ryan Fanello, Carlo Ciliberto, Giulia Pasquale
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

add_subdirectory(linearClassifierModule)
#add_subdirectory(sparseCoder)

option(BUILD_GIE "Build GIE" ON)
option(BUILD_caffe "Build caffe" ON)

if (BUILD_GIE)
	add_subdirectory(GIECoder)
endif()

if (BUILD_caffe)
	find_package(Caffe)
	if (Caffe_FOUND)
    		message(STATUS "Caffe is found => caffeCoder will be compiled")
    		add_subdirectory(caffeCoder)
	else()
    		message(STATUS "Caffe not found => skip compiling caffeCoder")
	endif()
endif()
