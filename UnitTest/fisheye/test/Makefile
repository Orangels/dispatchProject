CC = g++
CFLAGS = `pkg-config gflags --cflags --libs` -I/home/user/Program/ls-dev/dispatchProject/UnitTest/fisheye/test/IMV/include #gflags
LDFLAGS = -L/home/user/Program/ls-dev/dispatchProject/UnitTest/fisheye/test/IMV/lib -L/usr/lib/x86_64-linux-gnu/ -L/usr/local/lib/ -Wl,-rpath,/home/user/Program/ls-dev/dispatchProject/UnitTest/fisheye/test/IMV/lib
#LIBS = `pkg-config gflags glew glfw3 opencv --libs` -lIMV1 -ldl -pthread -lX11 #gflags
LIBS = -lgflags -lIMV1 -ldl -pthread -lX11 #gflags
CULIBS = -I/usr/local/cuda/include
#OPENCV = `pkg-config opencv --cflags` `pkg-config opencv --libs`
OPENCV = `pkg-config opencv --libs`

#SOFLAG = -shared -fPIC
#PROG = libdewarper.so #2019.12.13

#PROG = libdewarper_.so

SOFLAG = -Wall
PROG = ls_test

#SRCS = $(wildcard *.cpp)
#SRCS = camera_view.cpp dewarper.cpp
SRCS = test.cpp
cu_SRCS = $(wildcard *.cu)


$(PROG): $(SRCS)
	$(CC)  -o $(PROG) $(SOFLAG) $(SRCS) $(CFLAGS) $(LDFLAGS) $(LIBS) $(CULIBS) \
	-I/usr/include/opencv4 \
	 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio


.PHONY: clean

clean:
	rm -f $(PROG) split.o
