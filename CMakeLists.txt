# Copyright: 2014 iCub Facility
# Author: Sean Ryan Fanello, Carlo Ciliberto, Giulia Pasquale
# CopyPolicy: Released under the terms of the GNU GPL v2.0.
cmake_minimum_required(VERSION 3.5)
project(himrep)

find_package(ICUBcontrib REQUIRED)
find_package(YARP 3.5.1 REQUIRED COMPONENTS os sig dev math cv)
find_package(ICUB REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${ICUB_MODULE_PATH} ${ICUBCONTRIB_MODULE_PATH})

include(ICUBcontribHelpers)
include(ICUBcontribOptions)
icubcontrib_set_default_prefix()

add_subdirectory(modules)

icubcontrib_add_uninstall_target()
