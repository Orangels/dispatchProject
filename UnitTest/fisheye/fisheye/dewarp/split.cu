//
// Created by 李大冲 on 2019-09-18.
//

#include "split.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#define uchar unsigned char
#define uint32 unsigned long


void _tmp_cudasafe(cudaError_t code, const char *message, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error:%s. %s. In %s line %d\n", cudaGetErrorString(code),
                message, file, line);
//        exit(-1);
    }
}


__global__ void cudaTransform(void *output, int pitchOutput, int hOut,
                              uchar *input, int pitchInput, int channels,
                              int baseWidth, int baseHeight, float xRatio, float yRatio) {
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int YIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = YIndex / hOut, yIndex = YIndex % hOut;
    if (xIndex < pitchOutput and batch < 3) {

        int x = (int) (xRatio * xIndex);
        int y = (int) (yRatio * yIndex);

        uchar *a;
        uchar *b;
        uchar *c;
        uchar *d;
        float xDist, yDist, blue, red, green;

        // X and Y distance difference
        xDist = (xRatio * xIndex) - x;
        yDist = (yRatio * yIndex) - y;

        x += (baseWidth >> (batch & 2)) * batch;
        y += baseHeight * ((x / pitchInput + (batch >> 1)) & 1);
        x %= pitchInput;

        // Points
        a = input + (y * pitchInput + x) * channels;
        b = input + (y * pitchInput + (x + 1)) * channels;
        c = input + ((y + 1) * pitchInput + x) * channels;
        d = input + ((y + 1) * pitchInput + (x + 1)) * channels;
        //actually it's bgr->012
        // blue
        blue = (a[2]) * (1 - xDist) * (1 - yDist) + (b[2]) * (xDist) * (1 - yDist) + (c[2]) * (yDist) * (1 - xDist) +
               (d[2]) * (xDist * yDist) - 122.7717;//;//

        // green
        green = ((a[1])) * (1 - xDist) * (1 - yDist) + (b[1]) * (xDist) * (1 - yDist) + (c[1]) * (yDist) * (1 - xDist) +
                (d[1]) * (xDist * yDist) - 115.9465;

        // red
        red = (a[0]) * (1 - xDist) * (1 - yDist) + (b[0]) * (xDist) * (1 - yDist) + (c[0]) * (yDist) * (1 - xDist) +
              (d[0]) * (xDist * yDist) - 102.9801;
        //float
        int block = pitchOutput * hOut;
        float *p = (float *) output + batch * block * channels + (YIndex % hOut) * pitchOutput + xIndex;
        //char
        //int block = 1;
        //uchar *p = (uchar *) output + (YIndex * pitchOutput + xIndex) * channels;
        *(p + 0 * block) = red;  //(uchar) (red);  //
        *(p + 1 * block) = green;//(uchar) (green);//
        *(p + 2 * block) = blue; //(uchar) (blue); //
        //*(uint32 *) p = 0xff000000 | ((((int) red) << 16)) | ((((int) green) << 8)) | ((int) blue);
    }
}

splitProcess::splitProcess(int w, int h, int c, int outW, int outH, int slots, int pad) :
        input_w(w), input_h(h), input_c(c), pad(pad), subBatch(3), out_w(outW), out_h(outH), slots(slots) {
    base_w = w / 3 * 2, base_h = h / 2, n_slots = slots - 1, last_slots = slots - 1;
    imageByteLength = w * h * sizeof(uchar) * c;
    newImageByteLength = outW * outH * sizeof(float) * c * subBatch;
    newImageLength = outW * outH * c * subBatch;
    ratio_width = 1.0 * (base_w + pad) / out_w, ratio_height = 1.0 * base_h / out_h;

    _tmp_cudasafe(cudaMalloc((void **) &pixels_dev, newImageByteLength * slots),
             "New image allocation ", __FILE__, __LINE__);
    _tmp_cudasafe(cudaEventCreate(&start), "start allocation ", __FILE__, __LINE__);
    _tmp_cudasafe(cudaEventCreate(&stop), "stop allocation ", __FILE__, __LINE__);
    count = 0;
}

void splitProcess::get_refer() {
//    last_slots = (last_slots + 1) % slots;//逻辑上和输出保持绑定关系即可
    last_slots = n_slots;//保持固定的同步
    n_slots = (n_slots + 1) % slots;
    refer_dyn = (void *) ((float *) pixels_dev + last_slots * newImageLength);
//    std::cout << "get refer:" << last_slots << std::endl;
}

void splitProcess::run() {
//    n_slots = (n_slots + 1) % slots;
    out_Pixels_dyn = (void *) ((float *) pixels_dev + n_slots * newImageLength);

    // Start measuring time
    cudaEventRecord(start, 0);
    // grid
    dim3 grids((out_w + 31) / 32, (out_h * subBatch + 31) / 32);
    dim3 threads(32, 32);
    // Do the bilinear transform on CUDA device
    cudaTransform << < grids, threads >> > (out_Pixels_dyn, out_w, out_h,
            (uchar *) ipt_pixels_dyn, input_w, input_c, base_w, base_h,
            ratio_width, ratio_height);
//            1.0 * (base_w + pad) / out_w, 1.0 * base_h / out_h);
    // Stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    //test_sv();//save as mat
//    printf("Time for the kernel: %f ms\n", time);
}

splitProcess::~splitProcess() {
    // Free memory
    ipt_pixels_dyn = NULL;
    _tmp_cudasafe(cudaFree(pixels_dev), "free cuda output ", __FILE__, __LINE__);
    _tmp_cudasafe(cudaEventDestroy(start), "destory start ", __FILE__, __LINE__);
    _tmp_cudasafe(cudaEventDestroy(stop), "destory stop ", __FILE__, __LINE__);
}

void splitProcess::get_2p_Img(void *out_Image) {
//    cv::Size size = cv::Size(input_w, input_h);
//    cv::Mat src1 = cv::Mat(size, CV_8UC3);
//    std::cout << "sp w" << input_w << " H:" << input_h << "C:" << input_c << std::endl;
//    cudaMemcpy(src1.data, ipt_pixels_dyn, imageByteLength, cudaMemcpyDeviceToHost);
//    std::cout << "sp 101" << std::endl;
//    cv::imwrite("1001.jpg", src1);
//    printf("CUDA dst: [%p]", out_Image);
    _tmp_cudasafe(cudaMemcpy(out_Image, ipt_pixels_dyn, imageByteLength, cudaMemcpyDeviceToHost),
             "from device to host", __FILE__, __LINE__);
}

void splitProcess::test_sv() {
    cv::Size size = cv::Size(out_w, out_h * subBatch);
    cv::Mat dst(size, CV_8UC3);
    int sz = out_w * out_h * sizeof(uchar) * input_c * subBatch;
    _tmp_cudasafe(cudaMemcpy(dst.data, out_Pixels_dyn, sz,
                        cudaMemcpyDeviceToHost), "from device to host", __FILE__, __LINE__);
    if (!dst.isContinuous()) dst = dst.clone();
    //Save image
    std::string res = "sp100" + std::to_string(count++) + ".jpg";
    cv::imwrite(res, dst);
}


int testmain(int argc, char *argv[]) {
//int main(int argc, char *argv[]) {
    cv::Mat src = cv::imread("perimeter_1001.jpg", 3);
    //cv::Mat dst(src);
    int w = src.cols, h = src.rows, c = src.channels();
    int pad = 128 * 2, subbatch = 3, new_c = c;
    int base_w = w / 3 * 2, base_h = h / 2;
//    int out_w = 1280, out_h = 640;
    int out_w = 896, out_h = 896;

    cv::Size size = cv::Size(out_w, out_h * subbatch);
    cv::Mat dst(size, CV_8UC3);
    //https://blog.csdn.net/dcrmg/article/details/52294259
    // Get output image size
    int imageByteLength = w * h * sizeof(uchar) * c;
    int newImageByteLength = out_w * out_h * sizeof(uchar) * new_c * subbatch;

    std::cout << "size:" << size << "depth:" << src.depth() << std::endl;
    std::cout << "w:" << w << " H" << h << " c" << c << " sizeof(uchar):" << sizeof(uchar) << std::endl;
    std::cout << "neww:" << out_w << " newH" << out_h << " c" << new_c << " sizeof(uchar):" << sizeof(uchar)
              << std::endl;
    // Create pointer to device and host pixels
    //uchar *pixels = (uchar*)image->pixels;
    uchar *pixels_dyn;

    // Copy original image
    _tmp_cudasafe(cudaMalloc((void **) &pixels_dyn, imageByteLength), "Original image allocation ", __FILE__, __LINE__);
    _tmp_cudasafe(cudaMemcpy(pixels_dyn, src.data, imageByteLength, cudaMemcpyHostToDevice),
             "Copy original image to device ", __FILE__, __LINE__);

    // Allocate new image on DEVICE
    void *newPixels_dyn;
    //uchar *newPixels = (uchar *) malloc(newImageByteLength);

    _tmp_cudasafe(cudaMalloc((void **) &newPixels_dyn, newImageByteLength), "New image allocation ", __FILE__, __LINE__);
    std::cout << "macllloc 2" << std::endl;
    cudaEvent_t start, stop;
    _tmp_cudasafe(cudaEventCreate(&start), "start allocation ", __FILE__, __LINE__);
    _tmp_cudasafe(cudaEventCreate(&stop), "stop allocation ", __FILE__, __LINE__);

    float time;
    // Start measuring time
    cudaEventRecord(start, 0);

    // grid
    dim3 grids((out_w + 31) / 32, (out_h * subbatch + 31) / 32);
    dim3 threads(32, 32);
    std::cout << "grids:" << std::endl;
    // Do the bilinear transform on CUDA device
    cudaTransform << < grids, threads >> > (newPixels_dyn, out_w, out_h,
            pixels_dyn, w, c, base_w, base_h, 1.0 * (base_w + pad) / out_w, 1.0 * base_h / out_h);

    // Stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    std::cout << " out_w" << (4 >> (1 & 2)) << "w:" << (4 >> (0 & 2)) << " c" << (4 >> (2 & 2)) << " out_h:"
              << out_h << std::endl;
    std::cout << " out_w" << (1 >> 1) << "w:" << (0 >> 1) << " c" << (2 >> 1) << " out_h:" << out_h << std::endl;

    // Copy scaled image to host
//    _tmp_cudasafe(cudaMemcpy(dst.data, newPixels_dyn + newImageByteLength / subbatch, newImageByteLength / subbatch,
//                        cudaMemcpyDeviceToHost), "from device to host", __FILE__, __LINE__);
    _tmp_cudasafe(cudaMemcpy(dst.data, newPixels_dyn, newImageByteLength,
                        cudaMemcpyDeviceToHost), "from device to host", __FILE__, __LINE__);

    // Free memory
    cudaFree(pixels_dyn);
    cudaFree(newPixels_dyn);

    cudaEventElapsedTime(&time, start, stop);
    printf("Time for the kernel: %f ms\n", time);

    //Save image
    cv::imwrite("1001.jpg", dst);
    return 0;
}