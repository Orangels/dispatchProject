//
// Created by 李大冲 on 2019-10-10.
//

#ifndef INFER__PYTHON_ROUTE_H
#define INFER__PYTHON_ROUTE_H

#include <Python.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "../dewarp/dewarper.h"

#define STEP 10

class python_route {
public:
    python_route(float ratio_h, float ratio_w, int H, int W);

    ~python_route();

    void LoadModel(float ratio_h, float ratio_w, int H, int W);

    void RunModel(int, int, void *);

    void PythonPost(void *, void *, void *, int, int, bool toCanvas,
                    cv::Mat ResImg, int media_id, int frame_id, const char *mac);

    void ParseRet(cv::Mat, float, float, deWarp *deobj = nullptr);

//    void SendDB(cv::Mat, int, int, const char *);

    void PythonInfer(int batch, int row, int col, void *ipt);

private:
    PyObject *pModule, *pFunc, *postFunc, *selfPost;//, *sendFunc
    std::vector <uchar> buffer;
};


//template<typename T>
//T *expand_line(T *input_, int n_points, int &idx, const int total = 100, bool closed = false);
//int *expand_line(float *input_, int n_points, int &idx, const int total, bool closed);
//void expand_line(float *input_, int *output_, int n_points, int &idx, const int total, bool closed);
//
//void expand_box(float *input_, std::vector<int> &output_, int &idx, const int total);
void expand_line(float *input_, int n_points, std::vector<int> &output_);

void expand_box(float *input_, std::vector<float> &output_);

#endif //INFER__PYTHON_ROUTE_H
