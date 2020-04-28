//
// Created by 李大冲 on 2019-09-18.
//

#ifndef DEWARPER_SPLIT_H
#define DEWARPER_SPLIT_H

#include <cuda_runtime.h>
//#include <opencv2/opencv.hpp>


class splitProcess {
public:
    splitProcess(int w, int h, int c, int outW, int outH, int slots = 1, int pad = 128);

    ~splitProcess();

    void run();

    void get_2p_Img(void *);

    void get_refer();

    void test_sv();

    void *ipt_pixels_dyn, *out_Pixels_dyn, *refer_dyn;
    float ratio_width,ratio_height;
private:
    int pad, subBatch, slots, n_slots, last_slots;
    void *pixels_dev;

    int input_w, input_h, input_c, count;
    int base_w, base_h, out_w, out_h;
    int imageByteLength, newImageByteLength, newImageLength;

    //measuring time
    cudaEvent_t start, stop;
    float time;
};


#endif //DEWARPER_SPLIT_H
