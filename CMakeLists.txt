# Copyright: 2013 iCub Facility
# Author: Sean Ryan Fanello, Carlo Ciliberto
# CopyPolicy: Released under the terms of the GNU GPL v2.0.
cmake_minimum_required(VERSION 2.6)
project(himrep)

find_package(YARP REQUIRED)
find_package(ICUB REQUIRED)
find_package(ICUBcontrib REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${YARP_MODULE_PATH} ${ICUB_MODULE_PATH} ${ICUBCONTRIB_MODULE_PATH})

include(YarpInstallationHelpers)
include(ICUBcontribHelpers)
include(ICUBcontribOptions)
icubcontrib_set_default_prefix()

add_subdirectory(linearClassifierModule)
add_subdirectory(sparseCoder)

icubcontrib_add_uninstall_target()

