CC = g++
CFLAGS = `pkg-config gflags glew glfw3 opencv --cflags` -I/home/user/project/dewarp/dewarper/gpu_cu/IMV/include #gflags
LDFLAGS = -L/home/user/project/dewarp/dewarper/gpu_cu/IMV/lib -Wl,-rpath,/home/user/project/dewarp/dewarper/gpu_cu/IMV/lib
LIBS = `pkg-config gflags glew glfw3 opencv --libs` -lIMV1 -ldl -pthread -lX11 #gflags
CULIBS = -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcuda
OPENCV = `pkg-config opencv --cflags` `pkg-config opencv --libs`

SOFLAG = -shared -fPIC
#PROG = libdewarper.so #2019.12.13
PROG = libdewarper_.so
#SOFLAG = -Wall
#PROG = dewarper_HAHA

#SRCS = $(wildcard *.cpp)
SRCS = camera_view.cpp dewarper.cpp
cu_SRCS = $(wildcard *.cu)


$(PROG): $(SRCS) split.o
	$(CC)  -o $(PROG) $(SOFLAG) split.o $(SRCS) $(CFLAGS) $(LDFLAGS) $(LIBS) $(CULIBS)

split.o: split.cu
	nvcc  -shared -Xcompiler -fPIC -c split.cu $(CULIBS) $(OPENCV)

.PHONY: clean

clean:
	rm -f $(PROG) split.o
