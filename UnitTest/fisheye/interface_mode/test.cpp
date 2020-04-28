#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <gflags/gflags.h>
#include "IMV1.h"
#include "cameraHandler.h"
#include <thread>
#include <future>
#include <sys/time.h>

DEFINE_string(lens, "vivotek", "camera lens");
//DEFINE_string(mode, "center", "dewarping mode");
DEFINE_string(mode, "perimeter", "dewarping mode");
DEFINE_string(input, "rtsp://root:admin123@192.168.88.27/live.sdp", "input video file name");
//DEFINE_string(input, "rtsp://root:admin123@192.168.88.26/live.sdp", "input video file name");
DEFINE_string(output, "perimeter_ls_test.mp4", "output video file name");

DEFINE_double(camera_rotation, 0.0, "camera rotation degree");
DEFINE_double(pixels_per_degree, 16.0, "pixels per degree");
DEFINE_double(center_zoom_angle, 90.0, "center zoom field of view");
DEFINE_double(perimeter_top_angle, 90.0, "perimeter top angle");
DEFINE_double(perimeter_bottom_angle, 30.0, "perimeter bottom angle");

using namespace std;
using namespace cv;


int64_t getCurrentTime()
{
    struct timeval tv;
    gettimeofday(&tv,NULL);    //该函数在sys/time.h头文件中
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}


cv::Size CalculateSize(string mode="center") {
    if (mode == "center") {
        int length = int(FLAGS_pixels_per_degree *
                         FLAGS_center_zoom_angle / 4) << 2;
        return cv::Size(length, length);
    } else {
        int width = int(FLAGS_pixels_per_degree * 180.0f / 4) << 2;
        int height = int(FLAGS_pixels_per_degree *
                         (FLAGS_perimeter_top_angle - FLAGS_perimeter_bottom_angle)
                         / 2) << 2;
        return cv::Size(width, height);
    }
}

int main(int argc, char *argv[]){
    cv::Size size_center = CalculateSize("center");
    cv::Size size_perimeter = CalculateSize("perimeter");
    cv::Mat src;
    cv::Mat dst_center(size_center, CV_8UC3);
    cv::Mat dst_perimeter(size_perimeter, CV_8UC3);
    cv::Mat dst_perimeter_resize(cv::Size(size_perimeter.width / 2, size_perimeter.height / 2), CV_8UC3);

    cv::VideoCapture cap(FLAGS_input.c_str());

    if (!cap.isOpened()) {
        std::cerr << "Unable to open video file for capturing" << std::endl;
        return -1;
    }

    int fourcc = int(cap.get(cv::CAP_PROP_FOURCC));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int frame_count = int(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int current_frame = 0, width = 1280, height = 640;
    cap.read(src);

    Mat resize_src;
    resize(src, resize_src, cv::Size(src.cols, src.rows));

    cout << src.cols << "  " << src.rows << endl;
    cout << dst_perimeter.cols << "  " << dst_perimeter.rows << endl;
    cout << dst_center.cols << "  " << dst_center.rows << endl;

    cameraHandler* camera_center = new cameraHandler(src.cols, src.rows, src.data,
                                                  dst_center.cols, dst_center.rows, dst_center.data,
                                                  FLAGS_center_zoom_angle);

//    cameraHandler* camera_center = new cameraHandler(src.cols, src.rows, src.data,
//                                                     dst_perimeter_resize.cols, dst_perimeter_resize.rows, dst_perimeter_resize.data,
//                                                        FLAGS_perimeter_top_angle,
//                                                        FLAGS_perimeter_bottom_angle);

    cameraHandler* camera_perimeter = new cameraHandler(src.cols, src.rows, src.data,
                                                     dst_perimeter.cols, dst_perimeter.rows, dst_perimeter.data,
                                                     FLAGS_perimeter_top_angle,
                                                     FLAGS_perimeter_bottom_angle);



    imwrite("perimeter_1.jpg", dst_perimeter);
    imwrite("center_1.jpg", dst_center);
    int frame_num = 1000;
    int64_t sum = 0;
    while (cap.read(src)){
        if (frame_num > 1200) break;
        frame_num++;

        auto perimeter_reader = [&](){
            int64_t start = getCurrentTime();
            camera_perimeter->run(src.data);
            int64_t end = getCurrentTime();
            cout << "perimeter frame num : " << frame_num << " cost : " << end-start << endl;
        };
        auto center_reader = [&](){
            int64_t start = getCurrentTime();
            camera_center->run(src.data);
            int64_t end = getCurrentTime();
            cout << "center frame num : " << frame_num << " cost : " << end-start << endl;
        };


        int64_t start = getCurrentTime();
        thread thread_perimeter(perimeter_reader);
        thread thread_center(center_reader);

        thread_perimeter.join();
        thread_center.join();

//        Mat dst_perimeter_resize_2p;
//        resize(dst_perimeter_resize, dst_perimeter_resize_2p, size_perimeter);

        int64_t end = getCurrentTime();
        sum += (end-start);
        cout << "frame num : " << frame_num << " cost : " << end-start << endl;

        string img_path, img_path_center, img_path_ori;
        img_path = "./imgs/" + to_string(frame_num) + ".jpg";
        img_path_center = "./imgs_center/" + to_string(frame_num) + ".jpg";
        img_path_ori = "./imgs_ori/" + to_string(frame_num) + ".jpg";
        cv::imwrite(img_path, dst_perimeter);
        cv::imwrite(img_path_center, dst_center);
//        cv::imwrite(img_path_ori, dst_perimeter_resize);
    }
    cout << "avg time cost : " << sum/200 << endl;


    return 0;
}