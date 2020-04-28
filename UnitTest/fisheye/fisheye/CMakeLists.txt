# cmake  -DCMAKE_CUDA_FLAGS="--expt-extended-lambda -std=c++11" ..
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(infer_ LANGUAGES CXX)
#set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CUDA_FLAGS "--expt-extended-lambda -std=c++11")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -pthread")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -std=c++11 -O0 -Wall -g -ggdb -pthread ")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -std=c++11 -O3 -Wall -pthread ")
find_package(CUDA REQUIRED)
find_package(GFLAGS REQUIRED)#gflags
#find_package(glew REQUIRED)
#find_package(glfw3 REQUIRED)#glfw3
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(OpenCV REQUIRED)
enable_language(CUDA)

find_package(GLEW REQUIRED)
if (GLEW_FOUND)
    include_directories(${GLEW_INCLUDE_DIRS})
    link_libraries(${GLEW_LIBRARIES})
endif ()
include_directories(${CUDA_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/csrc/
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/ ${gflags_INCLUDE_DIR} #${CMAKE_CURRENT_SOURCE_DIR}/dewarp/
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/plugins/ ${OpenCV_INCLUDE_DIRS} /usr/local/include/
        ${CMAKE_CURRENT_SOURCE_DIR}/dewarp ${CMAKE_CURRENT_SOURCE_DIR}/dewarp/IMV/include/
        ${PYTHON_INCLUDE_DIRS} /home/user/Downloads/cub-1.8.0)

#include_directories(${CUDA_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/csrc/ ${OpenCV_INCLUDE_DIRS})
message("cu_I" ,${CUDA_INCLUDE_DIRS})
message("cu_L" ,${CUDA_LIBRARIES})
message("cur2" ,${PROJECT_SOURCE_DIR})

file(GLOB cudaSources ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/decode.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/decode_fcos.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/nms.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/groupnorm.cu)
file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/csrc/engine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/engine_fcos.cpp)
file(GLOB dewarpSources ${CMAKE_CURRENT_SOURCE_DIR}/dewarp/camera_view.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/dewarp/dewarper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/dewarp/split.cu)

#add_executable(export ${CMAKE_CURRENT_SOURCE_DIR}/export.cpp ${sources} ${cudaSources})
#target_link_libraries(export PRIVATE ${OpenCV_LIBS} ${CUDA_LIBRARIES} nvinfer nvonnxparser)
add_executable(export inferbase/export_fcos.cpp ${sources} ${cudaSources})
target_link_libraries(export PRIVATE ${OpenCV_LIBS} ${CUDA_LIBRARIES} nvinfer nvonnxparser)

#add_library(infer_ SHARED ${CMAKE_CURRENT_SOURCE_DIR}/infer_.cpp ${sources} ${cudaSources})
add_executable(infer_ inferbase/infer_.cpp inferbase/python_route.cpp
        ${sources} ${cudaSources} ${dewarpSources})

target_link_libraries(infer_ PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dewarp/IMV/lib/libIMV1.so
	 GLEW glfw3 GL X11 Xi Xrandr Xxf86vm Xinerama Xcursor rt m dl cudart cuda gflags GLU util #dewarp
        ${OpenCV_LIBS} gflags python3.5m ${CUDA_LIBRARIES} nvinfer nvonnxparser)

add_library(infer_2 SHARED inferbase/infer_lib.cpp ${sources} ${cudaSources})
target_link_libraries(infer_2 PRIVATE ${OpenCV_LIBS} cuda ${CUDA_LIBRARIES} nvinfer nvonnxparser)

add_library(infer_test SHARED inferbase/infer_test.cpp ${sources} ${cudaSources})
target_link_libraries(infer_test PRIVATE ${OpenCV_LIBS} ${CUDA_LIBRARIES} nvinfer nvonnxparser)

# gflags url: http://www.yeolar.com/note/2014/12/14/gflags/
