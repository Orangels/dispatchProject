//
// Created by 李大冲 on 2019-08-25.
//

#ifndef RETINANET_INFER_INFER_2_H
#define RETINANET_INFER_INFER_2_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "../csrc/engine.h"
#include "../dewarp/dewarper.h"
#include "python_route.h"

class Infer_RT {
public:
    ~Infer_RT();

    Infer_RT(const char *engine_file, const char *input = NULL,
             int meida_id = 0, const char *mac = NULL, const char *yaml = NULL,
             int device = -1, std::string modes = "perimeter");

    //device是摄像头设备号
    void process_();

    template<typename T>
    void setInfo(T *ptr, const char *input, int device, std::string modes);

    void preprocess(bool joinAll = false, bool cpdst = false);

    void postprocess(int, bool, bool &);

    void _getdst(int, int);

    void decorate();

    void _fowShow();

    void getsrc();

    void checkStop();

    static void *process(void *);

    void run();

//    void cal_src_ploygon(cv::Mat src, cv::Mat dst);

    deWarp *dewarper;
    cv::Mat src;
    std::vector <cv::Mat> showDsts;
private:
    def_retinanet::Engine *engine;

    void *data_d, *scores_d, *boxes_d, *classes_d;

    int channels, num_det, height, width, slots;

    int run_batch, N, N_s, N_b, n_count, n_post;

    int meida_id, dst_curid = 1, cur = 1, send_cur;

    const char *mac;//, *yaml

    float h_ratio, w_ratio, show_ratio_h, show_ratio_w;

    float *scores, *boxes, *classes;

    python_route *pr;

    bool stop, testStar, baseOnSrc;

    std::string yaml, line;

    cv::Size ResImgSiz;
};

#endif //RETINANET_INFER_INFER_2_H
