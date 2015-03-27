# Copyright: 2014 iCub Facility
# Author: Sean Ryan Fanello, Carlo Ciliberto, Giulia Pasquale
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

add_subdirectory(linearClassifierModule)
add_subdirectory(sparseCoder)

find_package(Caffe)
if (Caffe_FOUND)
    message(STATUS "Caffe is found => caffeCoder will be compiled")
    add_subdirectory(caffeCoder)
else()
    message(STATUS "Caffe not found => skip compiling caffeCoder")
endif()
