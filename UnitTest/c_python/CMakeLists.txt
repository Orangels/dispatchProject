cmake_minimum_required(VERSION 3.5)
project(c_python)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -g")

find_package(OpenCV REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        /usr/include/python3.5m
        /usr/local/lib/python3.5/dist-packages/numpy/core/include
)

link_libraries(
        /usr/lib/x86_64-linux-gnu/libpython3.5m.so
        ${OpenCV_LIBS}
)

add_executable(c_python main.cpp EnginePy.cpp mat2numpy.cpp)