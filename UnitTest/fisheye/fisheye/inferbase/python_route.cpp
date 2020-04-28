//
// Created by 李大冲 on 2019-10-10.
//

//https://blog.csdn.net/ziweipolaris/article/details/83689597
// g++ -o cc cc.cpp  `pkg-config --cflags --libs opencv` -I/usr/include/python3.5 -lpython3.5m
//https://stackoverflow.com/questions/9826311/trying-to-understand-linking-procedure-for-writing-python-c-hybrid
//opencv_demo.cpp
// how to use python class & function in c++: https://blog.csdn.net/sihai12345/article/details/82745350
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION //需要放在numpy/arrayobject.h之前
#define Mmax(a, b)(a)>(b)?(a):(b)

#include "python_route.h"
//#include <opencv/cv.hpp>
#include <Python.h>
#include <iostream>
#include <vector>
#include <numpy/arrayobject.h>
#include <opencv2/opencv.hpp>
//#include "base64.h"
#include <exception>

size_t init() {
    import_array();
}

python_route::~python_route() {
    Py_DECREF(pModule);
    Py_DECREF(pFunc);
    Py_DECREF(postFunc);
    Py_DECREF(selfPost);
    Py_Finalize();
}

python_route::python_route(float ratio_h, float ratio_w, int H, int W) {
    //int argc, char *argv[]
    //wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    //if (program == NULL) {
    //    fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
    //    exit(1);
    //}
    //Py_SetProgramName(program);  /* 不见得是必须的 */
    /* 非常重要，折腾的时间主要是因为这儿引起的【1】 */
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\"/home/user/project/fisheye/py_extension\")");
//    PyRun_SimpleString("sys.path.append(\"/srv/fisheye_prj/AI_Server/utils/py_extension\")");

    init();
    LoadModel(ratio_h, ratio_w, H, W);
}

void python_route::LoadModel(float ratio_h, float ratio_w, int H, int W) {
    /* 导入模块和函数，貌似两种方式都可以，不需要加.py，后面回再提到 */
    // PyObject *pName = PyUnicode_DecodeFSDefault("simple_module");
    PyObject *pName = PyUnicode_FromString("simple_module");
    /*这些检查也非常有帮助*/
    if (pName == NULL) {
        PyErr_Print();
        throw std::invalid_argument("Error: PyUnicode_FromString");
    }
    pModule = PyImport_Import(pName);
    if (pModule == NULL) {
        PyErr_Print();
        throw std::invalid_argument("fails to import the module");
    }
    pFunc = PyObject_GetAttrString(pModule, "simple_func");
    if (pFunc == NULL) {
        PyErr_Print();
        throw std::invalid_argument("fails to PyObject_GetAttrString");
    }

    postFunc = PyObject_GetAttrString(pModule, "box_info");
    //sendFunc = PyObject_GetAttrString(pModule, "sendToDatabase");
    PyObject *setFunc = PyObject_GetAttrString(pModule, "set_param");
    if (postFunc == NULL or setFunc == NULL) {//or sendFunc == NULL
        PyErr_Print();
        throw std::invalid_argument("fails to build python instance");
    }
    PyObject *pArgs = PyTuple_New(4);
    PyTuple_SetItem(pArgs, 0, Py_BuildValue("f", ratio_h));
    PyTuple_SetItem(pArgs, 1, Py_BuildValue("f", ratio_w));
    PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", H));
    PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", W));
    PyObject *pRetValue = PyObject_CallObject(setFunc, pArgs);

    // Py_DECREF
    Py_DECREF(pName);
    Py_DECREF(setFunc);
    Py_DECREF(pArgs);
    Py_DECREF(pRetValue);
}

//void python_route::SendDB() {
//
//    PyObject_CallObject(sendFunc, pArgs);
//    Py_DECREF(pArgs);
//}

void python_route::PythonPost(void *boxes, void *scores, void *classes, int run_batch,
                              int num_det, bool toCanvas,
                              cv::Mat ResImg, int media_id, int frame_id, const char *mac) {
    PyObject *pArgs = PyTuple_New(7);
    npy_intp dims_s[] = {run_batch, num_det};
    PyObject *sValue = PyArray_SimpleNewFromData(2, dims_s, NPY_FLOAT32, scores);
    npy_intp dims_c[] = {run_batch, num_det};
    PyObject *cValue = PyArray_SimpleNewFromData(2, dims_c, NPY_FLOAT32, classes);
    npy_intp dims_b[] = {run_batch, num_det, 4};
    PyObject *bValue = PyArray_SimpleNewFromData(3, dims_b, NPY_FLOAT32, boxes);

    PyTuple_SetItem(pArgs, 0, sValue);
    PyTuple_SetItem(pArgs, 1, cValue);
    PyTuple_SetItem(pArgs, 2, bValue);

    //those are not send everytime mqtt
    if (!ResImg.empty()) {
        buffer.resize(static_cast<size_t>(ResImg.rows) * static_cast<size_t>(ResImg.cols));
        cv::imencode(".jpg", ResImg, buffer);
        auto *enc_msg = reinterpret_cast<unsigned char *>(buffer.data());
        npy_intp dims_m[] = {buffer.size(), 1};
        //https://blog.csdn.net/jacke121/article/details/78536432
        PyObject *mValue = PyArray_SimpleNewFromData(2, dims_m, NPY_UBYTE, enc_msg);
        PyTuple_SetItem(pArgs, 3, mValue);
        //std::cout << "sssss: " << buffer.size() << std::endl;
    } else {
        //std::cout << "ddddd: " << std::endl;
        PyTuple_SetItem(pArgs, 3, Py_BuildValue("s", ""));
    }
    PyTuple_SetItem(pArgs, 4, Py_BuildValue("i", media_id));
    PyTuple_SetItem(pArgs, 5, Py_BuildValue("i", frame_id));
    PyTuple_SetItem(pArgs, 6, Py_BuildValue("s", mac));

    if (toCanvas) {
        selfPost = PyObject_CallObject(postFunc, pArgs);
        if (selfPost == NULL) {
            PyErr_Print();
            throw std::invalid_argument("CalllObject return NULL");
        }
    } else {
        PyObject *pRetValue = PyObject_CallObject(postFunc, pArgs);
        Py_DECREF(pRetValue);
    }

    // Py_DECREF
    Py_DECREF(pArgs);
}

/**/
void python_route::ParseRet(cv::Mat dst, float ratiow, float ratioh, deWarp *deobj) {
    int half_h = dst.rows / 2;
    // 解析返回结果
    PyArrayObject *r1, *r2, *r3, *r4, *r5, *r6, *r7;
    if (!PyArg_UnpackTuple(selfPost, "ref", 7, 7, &r1, &r2, &r3, &r4, &r5, &r6, &r7)) {
        PyErr_Print();
        throw std::invalid_argument("PyArg_ParseTuple Crash!");
    }
    npy_intp *shape1 = PyArray_SHAPE(r1);
    npy_intp *shape2 = PyArray_SHAPE(r2);
    npy_intp *shape3 = PyArray_SHAPE(r3);
    npy_intp *shape4 = PyArray_SHAPE(r4);
    npy_intp *shape5 = PyArray_SHAPE(r5);
    npy_intp *shape6 = PyArray_SHAPE(r6);
    npy_intp *shape7 = PyArray_SHAPE(r7);
    if (!shape1[0]) return;

//    std::cout << "shape[1]:" << shape1[0] <<
//              " shape2:" << shape2[0] << "," << shape2[1] <<
//              " shape3:" << shape3[0] <<
//              " shape4:" << shape4[0] << "," << shape4[1] <<
//              " shape5:" << shape5[0] <<
//              " shape6:" << shape6[0] << "," << shape6[1] <<
//              " shape7:" << shape7[0] <<
//              std::endl;

    float *base_r1 = (float *) PyArray_DATA(r1);//身份id N
    float *base_r2 = (float *) PyArray_DATA(r2);//跟踪线 n*2
    float *base_r3 = (float *) PyArray_DATA(r3);//跟踪数据的长度 N
    float *base_r4 = (float *) PyArray_DATA(r4);//人体框 N*4
    float *base_r5 = (float *) PyArray_DATA(r5);//进出路过3个数字 3
    float *base_r6 = (float *) PyArray_DATA(r6);//进出线框，4*2
    float *base_r7 = (float *) PyArray_DATA(r7);//颜色对应id，(3*N)
    std::vector<float> ids(base_r1, base_r1 + shape1[0]);
    //输入mat是原图或2p图时，这三个需要不同定义
    std::vector<float> track, track_num, box;
    std::vector<float> Box_line, Box_line_num;
    std::vector<float> statistic(base_r5, base_r5 + shape5[0]);
    std::vector<float> support(base_r6, base_r6 + shape6[0] * shape6[1]);
    std::vector<float> color(base_r7, base_r7 + shape7[0]);
    if (deobj != nullptr) {
        int base_num = 0;
        for (auto i = 0; i < shape1[0]; i++) {
            int num = int(base_r3[i]), total_line = num * shape2[1], total_box;
            float *tmp = base_r2 + base_num * shape2[1];
            std::vector<float> poloyline, poloygons;

            //line
            poloyline.insert(poloyline.end(), tmp, tmp + num * shape2[1]);
            float output_[total_line];
            deobj->mappingPolygon(poloyline, output_);
            track.insert(track.end(), output_, output_ + total_line);
            track_num.push_back(num);
            base_num += num;

            //box
            tmp = base_r4 + i * shape4[1];
            expand_box(tmp, poloygons);
            total_box = int(poloygons.size());
            float b_output_[total_box];
            deobj->mappingPolygon(poloygons, b_output_);
            Box_line.insert(Box_line.end(), b_output_, b_output_ + total_box);
            Box_line_num.push_back(total_box / 2);
        }
    } else {
        track.insert(track.end(), base_r2, base_r2 + shape2[0] * shape2[1]);
        track_num.insert(track_num.end(), base_r3, base_r3 + shape3[0]);
    }
    box.insert(box.end(), base_r4, base_r4 + shape4[0] * shape4[1]);

    int cw = shape1[0] > 0 ? shape7[0] / shape1[0] : 3;
    //track line;画图
    for (auto i = 0, j = 0, m = 0; i < shape3[0]; i++) {
        if (i > track_num.size() or i > box.size())break;
        int len = int(track_num[i]), id = int(ids[i]);
        int r = int(color[cw * i]), g = int(color[cw * i + 1]), b = int(color[cw * i + 2]);
        float x1 = round(box[4 * i]), y1 = round(box[4 * i + 1]),
                w = round(box[4 * i + 2]), h = round(box[4 * i + 3]);
        int ratio = int(1.0 * w * h / (50 * 100)), style = 2;
        ratio = ratio < 5 ? ratio : 5;
        style = style > ratio ? ceil(style * ratioh) : ceil(ratio * ratioh);


        for (auto k = j; k < j + len; k++) {
            //std::cout << cv::Point(track[2 * k], track[2 * k + 1]) << ';';
            if (2 * k > track.size())break;
            if (k == j) {
                cv::circle(dst, cv::Point(track[2 * k] * ratiow, track[2 * k + 1] * ratioh),
                           ceil(8 * ratioh), cv::Scalar(r, g, b), -1);
                continue;
            }
            float status = (track[2 * k - 1] * ratiow - half_h) * (track[2 * k + 1] * ratioh - half_h);
            if (status < 0)continue;
            cv::line(dst, cv::Point(track[2 * k - 2] * ratiow, track[2 * k - 1] * ratioh),
                     cv::Point(track[2 * k] * ratiow, track[2 * k + 1] * ratioh), cv::Scalar(r, g, b), style);
        }
        j += len;
        if (deobj != nullptr) {
            len = int(Box_line_num[i]);
            for (auto k = m; k < m + len; k++) {
                if (k == m)continue;
                if (2 * k > Box_line.size())break;
                cv::line(dst, cv::Point(Box_line[2 * k - 2] * ratiow, Box_line[2 * k - 1] * ratioh),
                         cv::Point(Box_line[2 * k] * ratiow, Box_line[2 * k + 1] * ratioh), cv::Scalar(r, g, b), style);
            }
            cv::putText(dst, std::to_string(id),
                        cv::Point(int(Box_line[2 * m] * ratiow) + 6, int(Box_line[2 * m + 1] * ratioh) + 6),
                        cv::FONT_HERSHEY_PLAIN, Mmax(2.0, style * 0.8), cv::Scalar(r, g, b), int(style * 0.8));
            m += len;
        } else {
            cv::putText(dst, std::to_string(id), cv::Point(int(x1 * ratiow) + 6, int(y1 * ratioh) + 6),
                        cv::FONT_HERSHEY_PLAIN, Mmax(2.0, style * 0.8), cv::Scalar(r, g, b), int(style * 0.8));
            cv::rectangle(dst, cv::Point(x1 * ratiow, y1 * ratioh),
                          cv::Point((x1 + w) * ratiow, (y1 + h) * ratiow), cv::Scalar(r, g, b), style);
        }
//        std::cout << "c:" << r << ", " << g << ", " << b << ";";
    }
    // Py_DECREF(selfPost);


//    std::cout << std::endl;

//    std::cout << "ids ";
//    for (auto c: ids)std::cout << c << ',';
//    std::cout << std::endl;
//    std::cout << "track ";
//    for (auto c: track)std::cout << c << ',';
//    std::cout << std::endl;
//    std::cout << "track_num ";
//    for (auto c: track_num)std::cout << c << ',';
//    std::cout << std::endl;
//    std::cout << "box ";
//    for (auto c: box)std::cout << c << ',';
//    std::cout << std::endl;
//    std::cout << "statistic ";
//    for (auto c: statistic)std::cout << c << ',';
//    std::cout << std::endl;
//    std::cout << "color ";
//    for (auto c: color)std::cout << c << ',';
//    std::cout << std::endl;

}

/**/

void python_route::PythonInfer(int batch, int row, int col, void *ipt) {
    /* 准备输入参数 */
    PyObject *pArgs = PyTuple_New(2);
    npy_intp dims[] = {batch, row, col, 3};

    PyObject *pValue = PyArray_SimpleNewFromData(4, dims, NPY_FLOAT32, ipt);
    PyTuple_SetItem(pArgs, 0, pValue);  /* pValue的引用计数被偷偷减一，无需手动再减 */
    PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", 2));    /* 图像放大2倍 */
    /* 调用函数 */
    PyObject *pRetValue = PyObject_CallObject(pFunc, pArgs);
    if (pRetValue == NULL) {
        PyErr_Print();
        throw std::invalid_argument("PythonInfer NULL");
    }
}

void python_route::RunModel(int row, int col, void *ipt) {
    /* 准备输入参数 */
    PyObject *pArgs = PyTuple_New(2);
    npy_intp dims[] = {row, col, 3};
    std::cout << "python 70" << std::endl;
    PyObject *pValue = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, ipt);
    PyTuple_SetItem(pArgs, 0, pValue);  /* pValue的引用计数被偷偷减一，无需手动再减 */
    PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", 2));    /* 图像放大2倍 */
    std::cout << "python 74" << std::endl;
    /* 调用函数 */
    PyObject *pRetValue = PyObject_CallObject(pFunc, pArgs);
    /* 解析返回结果 */
    PyArrayObject *ret_array;
    std::cout << "python 79" << std::endl;
    PyArray_OutputConverter(pRetValue, &ret_array);
    std::cout << "python 81" << std::endl;
    npy_intp *shape = PyArray_SHAPE(ret_array);
    cv::Mat big_img(shape[0], shape[1], CV_8UC3, PyArray_DATA(ret_array));
    cv::imwrite("aa.jpg", big_img);
}

int pymain(int argc, char *argv[]) {
    /* 读图 */
    cv::Mat sml_img = cv::imread("build/1001.jpg");
    if (!sml_img.isContinuous()) { sml_img = sml_img.clone(); }
    int row = sml_img.rows, col = sml_img.cols;
    python_route pr = python_route(0, 0, 0, 0);
    pr.RunModel(row, col, sml_img.data);
    return 0;
}


//template<typename T>int &idx, const int total,
void expand_line(float *input_, int n_points, std::vector<int> &output_) {
    // "perimeter top angle",90.0; perimeter_bottom_angle, 30.0;
    //是否为闭合曲线，需要区别对待
    //480是内圈半径，但不严格，用来度量最内圈应该用多大的步长。
    //setp是最外圈的步长，也就是在2p图上最小的像素步长，越靠近内圈步长越大
    int one_third_R = 480, step = STEP, i = 0;
    for (; i < n_points - 1; i++) {
        float delta_w = input_[2 * (i + 1)] - input_[2 * i];
        float delta_h = input_[2 * (i + 1) + 1] - input_[2 * i + 1];
        //all肯定不为0
        float all = sqrt((1.0 * pow(delta_h, 2) + 1.0 * pow(delta_w, 2)));
        if (all <= step) {
            output_.push_back(input_[2 * i]);
            output_.push_back(input_[2 * i + 1]);
            continue;
        }
        //错误是：会产生横跨2p图的线条；
//        //越接近中间线，步长越大
        float local_s = 1.0 * step * 3 * one_third_R / (abs(2 * one_third_R - input_[2 * i + 1]) + one_third_R);
        int N_step = ceil(all / local_s);
        float sin_ = 1.0 * delta_h / all, cos_ = 1.0 * delta_w / all, local_step = 1.0 * all / N_step;
//        std::cout << "locals:" << local_s << ',' << N_step<<','<<local_step << std::endl;
//        std::cout << "locals:" << input_[2 * i] << ',' <<input_[2 * i + 1] << std::endl;
        //闭合图形，不保留终点，只保留起始点
        for (int j = 0; j < N_step; ++j) {
            int x_ = int((j * local_step) * cos_ + input_[2 * i]);
            int y_ = int((j * local_step) * sin_ + input_[2 * i + 1]);
            output_.push_back(x_);
            output_.push_back(y_);
            if (cos_ >= -1e-2 and cos_ <= 1e-2) break;//认为在半径上是没有拉伸的,只加第一个点
        }
    }
    i = n_points - 1;
    output_.push_back(int(input_[2 * i]));
    output_.push_back(int(input_[2 * i + 1]));
}

void expand_box(float *input_, std::vector<float> &output_) {
    // "perimeter top angle",90.0; perimeter_bottom_angle, 30.0;
    //input: x1y1wh
    //480是内圈半径，但不严格，用来度量最内圈应该用多大的步长。
    //setp是最外圈的步长，也就是在2p图上最小的像素步长，越靠近内圈步长越大
//     vector<int>out_x,out_y,out_;
    int one_third_R = 480, step = STEP;
    //idx = 0;
    float x1 = round(input_[0]), y1 = round(input_[1]),
            w = round(input_[2]), h = round(input_[3]);
//    std::cout << "box:" << x1 << ',' << y1 << ',' << w << ',' << h << std::endl;

    float w_idx[] = {0, 0, 1, 1}, h_idx[] = {0, 1, 1, 0};
    //left bottom right top
    for (int i = 0; i < 4; i++) {
        float x_ = x1 + w * w_idx[i];
        float y_ = y1 + h * h_idx[i];
        if (i % 2 == 0) {
            output_.push_back(x_);
            output_.push_back(y_);
//            std::cout << "  " << x_ << ", " << y_ << std::endl;
        } else {
            //越接近中间线，步长越大 (3*step)* delta; delta = r/(r +|2r-h|); delta = [1/3, 1]
            float local_s = 1.0 * step * 3 * one_third_R / (abs(2 * one_third_R - y_) + one_third_R);
            int N_step = ceil(w / local_s);
            float local_step = (-1.0) * (i / 2) * w / N_step;
            //闭合图形，不保留终点，只保留起始点
            for (int j = 0; j < N_step; ++j) {
                output_.push_back(j * local_step + x_);
                output_.push_back(y_);
//                std::cout << "  " << j * local_step + x_ << ", " << y_ << std::endl;
            }
        }
//        std::cout << "box I:" << i << "box x:" << x_ << "box y:" << y_ << std::endl;
    }
    //add first one point again
    output_.push_back(x1);
    output_.push_back(y1);
}

