cmake_minimum_required(VERSION 2.8)
PROJECT(Tracking)

IF(CMAKE_BUILD_TYPE)
ELSE()
    SET(CMAKE_BUILD_TYPE "Debug")
ENDIF()

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
CONAN_BASIC_SETUP()

SET(CMAKE_CXX_FLAGS_DEBUG " $ENV{CXXFLAGS} -DDEBUG -g -w ")
SET(CMAKE_CXX_FLAGS_RELEASE " $ENV{CXXFLAGS} -std=c++11  -DNDEBUG -O3 -w ")
SET(PLATFORM ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR})

#add_compile_options(-std=c++11)
INCLUDE_DIRECTORIES(src include /usr/local/include /usr/local/cuda/include)
LINK_DIRECTORIES(/usr/local/lib /usr/lib lib /usr/local/cuda/lib64)

FIND_LIBRARY(LIB_CAFFE caffe HINTS ${CONAN_LIB_DIRS_CAFFE}/${PLATFORM})
FIND_LIBRARY(LIB_PROTO proto HINTS ${CONAN_LIB_DIRS_PROTO}/${PLATFORM})

message(${CONAN_LIB_DIRS_CAFFE}/${PLATFORM})
message(${CONAN_LIB_DIRS_PROTO}/${PLATFORM})



SET(LIB_OPENCV opencv_core opencv_highgui opencv_imgproc opencv_video)
LIST(APPEND LIBS cublas cudart glog boost_system ${LIB_OPENCV} ${LIB_CAFFE} ${LIB_PROTO})

ADD_SUBDIRECTORY(src)
