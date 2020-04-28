//
// Created by 李大冲 on 2019-09-02.
//

#ifndef RETINANET_INFER_DEWARPER_H
#define RETINANET_INFER_DEWARPER_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <vector>

//#include <gflags/gflags.h>
//#include "split.h"
//#include "camera_view.h"

/*
int run(uchar *input, int row, int col,
        uchar *output, int row_out, int col_out,
        int current_frame,
        const char *lens, const char *mode,
        int camera_rotation = 0,
        int center_zoom_angle = 90,
        int perimeter_top_angle = 95,
        int perimeter_bottom_angle = 30);
*/
class deWarp {
public:
    deWarp(int w = 1280, int h = 640, int slot = 2, std::string modes = "perimeter",
           bool saveVideo = false, bool savePhoto = false, std::string len = "vivotek",
           float camera_rotation = 0.0, float pixels_per_degree = 16.0,
           float center_zoom_angle = 90.0, float perimeter_top_angle = 90.0,
           float perimeter_bottom_angle = 30.0);

    ~deWarp();

    void process(bool cpdst = false, bool baseOnSrc = false);

    int test();

    bool checkStatus();

    void currentStatus();

    void currentImg();

    void saveImg(std::string, cv::Mat ResImg, bool savebase = false);

    void readVideo(const char *input = NULL, int device = -1);

//    static void *run(void *);
//
//    void lanch();
    void join_thread();

    void join_dst();

    void mappingPolygon(std::vector<float> &output_, float *input_);

    void *data;//, *Img
    cv::Mat src, dst, src_clone;
    int rows, cols, row_out, col_out, current_frame;
    float ratio_w, ratio_h;
    bool has_frame;
private:
    std::vector <cv::Mat> dsts;
    std::vector<bool> status;
    int width, height;
    int fourcc, frame_count, format;
    std::string lens, mode;
    float camera_rotation, center_zoom_angle, pixels_per_degree;
    float perimeter_top_angle, perimeter_bottom_angle;
    cv::VideoCapture cap;
    const char *rpl;
    void *camera;
    double fps;
    int slots, n_slots, last_slots, dst_slots;
    cv::Size srcSize;
    cv::VideoWriter writer;
    bool save_video, save_photo, init_, readSwitch;
    std::thread readsrc, getdst;
    std::queue <cv::Mat> mul_mat;
    int extraDst;
};

#endif //RETINANET_INFER_DEWARPER_H
