cmake_minimum_required(VERSION 3.5)
project(test)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -g")

execute_process(COMMAND mkdir imgs)
execute_process(COMMAND mkdir imgs_center)

find_package(OpenCV REQUIRED)
find_package(GFLAGS REQUIRED)

message(${OpenCV_INCLUDE_DIRS})

include_directories(
        includes
        ${OpenCV_INCLUDE_DIRS}
        ${gflags_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/IMV/include/
)
link_libraries(
        ${OpenCV_LIBS}
        ${CMAKE_CURRENT_SOURCE_DIR}/IMV/lib/libIMV1.so
        gflags

)

add_executable(test test.cpp cameraHandler.cpp)
