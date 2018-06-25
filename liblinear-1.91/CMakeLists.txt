CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

SET(PROJECTNAME Liblinear)
PROJECT(${PROJECTNAME})

include_directories(${PROJECT_SOURCE_DIR})


FILE(GLOB folder_source *.cpp  *.c )
FILE(GLOB folder_header *.h  blas/*.h)

SOURCE_GROUP("Source Files" FILES ${folder_source})
SOURCE_GROUP("Header Files" FILES ${folder_header})

add_library(${PROJECTNAME} ${folder_source} ${folder_header})