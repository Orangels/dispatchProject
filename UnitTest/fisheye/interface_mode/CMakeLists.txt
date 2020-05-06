cmake_minimum_required(VERSION 3.5)
project(test)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -g")

execute_process(COMMAND mkdir imgs)
execute_process(COMMAND mkdir imgs_center)
execute_process(COMMAND mkdir imgs_ori)

find_package(OpenCV REQUIRED)
find_package(GFLAGS REQUIRED)
find_package(GLEW REQUIRED)
find_package(CUDA REQUIRED)

if (GLEW_FOUND)
    message(found glew)
    message(${GLEW_INCLUDE_DIRS})
    include_directories(${GLEW_INCLUDE_DIRS})
    link_libraries(${GLEW_LIBRARIES})
endif ()

message(${OpenCV_INCLUDE_DIRS})

file(GLOB gcamera ${CMAKE_CURRENT_SOURCE_DIR}/fishCamHandler/*cpp)

include_directories(
        includes
        ${OpenCV_INCLUDE_DIRS}
        ${gflags_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/IMV/include/
        ${CUDA_INCLUDE_DIRS}
)
link_libraries(
        ${OpenCV_LIBS}
        ${CMAKE_CURRENT_SOURCE_DIR}/IMV/lib/libIMV1.so
        gflags
        GLEW glfw3 GL
        X11 Xrandr Xxf86vm Xinerama Xcursor rt m dl
#        Xi Xrandr Xxf86vm Xinerama Xcursor rt m dl
        cuda
        ${CUDA_LIBRARIES}

)



#add_executable(ls_test main.cpp cameraHandler.cpp )
add_executable(ls_test_g test.cpp ${gcamera})
