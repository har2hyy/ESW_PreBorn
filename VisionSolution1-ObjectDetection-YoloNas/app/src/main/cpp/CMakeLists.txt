# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

cmake_minimum_required(VERSION 3.18.1)
project("objectdetectionYoloNas")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
set(OpenCV_DIR ${PROJECT_ROOT}/sdk/native/jni)
set(SNPE_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI})

find_package(OpenCV REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc/zdl)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc/hpp)

add_library(objectdetectionYoloNas
        SHARED
        objectdetectionYoloNas.cpp
        inference.cpp
        inference_helper.cpp)

find_library(log-lib log)
find_library(android-lib android)

set(SNPE_SHARED_LIBS
        ${SNPE_LIB_DIR}/libSNPE.so
        ${SNPE_LIB_DIR}/libsnpe-android.so)

target_link_libraries(objectdetectionYoloNas
        ${OpenCV_LIBS}
        ${SNPE_SHARED_LIBS}
        ${android-lib}
        ${log-lib})
