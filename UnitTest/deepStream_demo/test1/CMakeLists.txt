cmake_minimum_required(VERSION 3.5.1)
project(ds_test1)

find_package(OpenCV REQUIRED)

find_package(PkgConfig)
pkg_search_module(GST REQUIRED gstreamer-1.0)
MESSAGE(STATUS ${GST_LIBRARIES})
MESSAGE(STATUS ${GST_INCLUDE_DIRS})


include_directories(/usr/include ${OpenCV_INCLUDE_DIRS}  /usr/local/include
        ${GST_INCLUDE_DIRS}
        /home/nano/Downloads/deepstream_sdk_v4.0.2_jetson/sources/includes
        )
file(GLOB sources src/*.cpp)

add_executable(ds_test1 ${sources})

target_link_libraries(ds_test1 ${OpenCV_LIBS}
        #        gstreamer-1.0 gobject-2.0 glib-2.0
        ${GST_LIBRARIES}
        /opt/nvidia/deepstream/deepstream-4.0/lib/libnvdsgst_meta.so
        /opt/nvidia/deepstream/deepstream-4.0/lib/libnvds_meta.so
        /opt/nvidia/deepstream/deepstream-4.0/lib/libnvbufsurface.so

        )