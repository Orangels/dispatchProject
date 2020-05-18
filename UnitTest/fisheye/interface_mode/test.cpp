#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <gflags/gflags.h>
#include "IMV1.h"
#include "fishCamHandler/cameraHandler.h"
#include <thread>
#include <future>
#include <sys/time.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

DEFINE_string(lens, "vivotek", "camera lens");
//DEFINE_string(mode, "center", "dewarping mode");
DEFINE_string(mode, "perimeter", "dewarping mode");
DEFINE_string(input, "rtsp://root:admin123@192.168.88.67/live.sdp", "input video file name");
//DEFINE_string(input, "/home/user/Program/ls-dev/DPH_Server_ls/DPH_Server/data/fish_0.mp4", "input video file name");
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

//void DestroyGLContext() {
//    glfwDestroyWindow(glfwGetCurrentContext());
//    glfwTerminate();
//}

//static void error_callback(int error, const char* description)
//{
//    fprintf(stderr, "Error: %s\n", description);
//}
//
//bool InitGLContext() {
//    cout << "InitGLContext 000" << endl;
//
//    glfwSetErrorCallback(error_callback);
//    if (glfwInit() != GLFW_TRUE) {
//        return false;
//    }
//    cout << "InitGLContext 111" << endl;
//    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
//    GLFWwindow *window = glfwCreateWindow(640, 480, "", NULL, NULL);
//    if (!window) {
//        glfwTerminate();
//        return false;
//    }
//    cout << "InitGLContext 222" << endl;
//    glfwMakeContextCurrent(window);
//
//    glewExperimental = GL_TRUE;
//    if (glewInit() != GLEW_OK) {
//        glfwDestroyWindow(window);
//        glfwTerminate();
//        return false;
//    }
//    cout << "InitGLContext 333" << endl;
//    return true;
//}


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

void dewrapView(int mode){
    //    google::ParseCommandLineFlags(&argc, &argv, true);
    cv::Size size_center = CalculateSize("center");
    cv::Size size_perimeter = CalculateSize("perimeter");
    cv::Mat src;
    cv::Mat dst_center(size_center, CV_8UC3);
    cv::Mat dst_perimeter(size_perimeter, CV_8UC3);
    cv::Mat dst_perimeter_resize(cv::Size(size_perimeter.width / 2, size_perimeter.height / 2), CV_8UC3);

    cv::VideoCapture cap(FLAGS_input.c_str());

    if (!cap.isOpened()) {
        std::cerr << "Unable to open video file for capturing" << std::endl;
    }

    int fourcc = int(cap.get(cv::CAP_PROP_FOURCC));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int frame_count = int(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int current_frame = 0, width = 1280, height = 640;
//    cap.read(src);

//    cv::VideoWriter writer(FLAGS_output.c_str(), CV_FOURCC('M', 'P', '4', 'V'), 20, size_perimeter);

    Mat resize_src;
    resize(src, resize_src, cv::Size(src.cols, src.rows));

    cout << src.cols << "  " << src.rows << endl;
    cout << dst_perimeter.cols << "  " << dst_perimeter.rows << endl;
    cout << dst_center.cols << "  " << dst_center.rows << endl;

    cameraGHandler* camera;

//    0: gpu, 1:cpu
    if (mode == 0) {
        camera = new cameraGHandler(src.cols, src.rows, src.data,
                                    dst_perimeter.cols, dst_perimeter.rows, dst_perimeter.data,
                                    FLAGS_perimeter_top_angle,
                                    FLAGS_perimeter_bottom_angle);
    } else {
        camera = new cameraGHandler(src.cols, src.rows, src.data,
                                    dst_center.cols, dst_center.rows, dst_center.data,
                                    FLAGS_center_zoom_angle);
    }

    int frame_num = 1000;
    int64_t sum = 0;
    while (cap.read(src)){
        if (frame_num > 1010) break;
        frame_num++;

        int64_t start = getCurrentTime();
        camera->run(src.data);
        int64_t end = getCurrentTime();
        sum += (end-start);
        cout <<"mode : " << mode << " frame num : " << frame_num << " cost : " << end-start << endl;

        string img_path, img_path_center, img_path_ori;
        img_path = "./imgs/" + to_string(frame_num) + ".jpg";
        img_path_center = "./imgs_center/" + to_string(frame_num) + ".jpg";
        img_path_ori = "./imgs_ori/" + to_string(frame_num) + ".jpg";
        if (mode == 0) {
            cv::imwrite(img_path, dst_perimeter);
        } else {
            cv::imwrite(img_path_center, dst_center);
        }

    }
    cout << "avg time cost : " << sum/10 << endl;

//    DestroyGLContext();
//    writer.release();
    cap.release();
}

void dewrapViewTotal(){

    auto center_func = [&](){
        //    google::ParseCommandLineFlags(&argc, &argv, true);
        cv::Size size_center = CalculateSize("center");
        cv::Size size_perimeter = CalculateSize("perimeter");
        cv::Mat src = cv::imread("/home/user/Program/ls-dev/dispatchProject/UnitTest/fisheye/interface/imgs_ori/ori.dat");
        cv::Mat dst_center(size_center, CV_8UC3);
        cv::Mat dst_perimeter(size_perimeter, CV_8UC3);
        cv::Mat dst_perimeter_resize(cv::Size(size_perimeter.width / 2, size_perimeter.height / 2), CV_8UC3);

        cv::VideoCapture cap(FLAGS_input.c_str());

        if (!cap.isOpened()) {
            std::cerr << "Unable to open video file for capturing" << std::endl;
        }

        int fourcc = int(cap.get(cv::CAP_PROP_FOURCC));
        double fps = cap.get(cv::CAP_PROP_FPS);
        int frame_count = int(cap.get(cv::CAP_PROP_FRAME_COUNT));
        int current_frame = 0, width = 1280, height = 640;
//        cap.read(src);

//    cv::VideoWriter writer(FLAGS_output.c_str(), CV_FOURCC('M', 'P', '4', 'V'), 20, size_perimeter);

        Mat resize_src;
        resize(src, resize_src, cv::Size(src.cols, src.rows));

        cout << src.cols << "  " << src.rows << endl;
        cout << dst_perimeter.cols << "  " << dst_perimeter.rows << endl;
        cout << dst_center.cols << "  " << dst_center.rows << endl;


        cameraGHandler* camera = new cameraGHandler(src.cols, src.rows, src.data,
                                                    dst_center.cols, dst_center.rows, dst_center.data,
                                                    FLAGS_center_zoom_angle);

        int frame_num = 1000;
        int64_t sum = 0;
        while (cap.read(src)){
            if (frame_num > 1010) break;
            frame_num++;

            int64_t start = getCurrentTime();
            camera->run(src.data);
            int64_t end = getCurrentTime();
            sum += (end-start);
            cout <<"mode : 1 frame num : " << frame_num << " cost : " << end-start << endl;

            string img_path, img_path_center, img_path_ori;
            img_path = "./imgs/" + to_string(frame_num) + ".jpg";
            img_path_center = "./imgs_center/" + to_string(frame_num) + ".jpg";
            img_path_ori = "./imgs_ori/" + to_string(frame_num) + ".jpg";

            cv::imwrite(img_path_center, dst_center);


        }
        cap.release();

    };

    auto perimeter_func = [&](){
        //    google::ParseCommandLineFlags(&argc, &argv, true);
        cv::Size size_center = CalculateSize("center");
        cv::Size size_perimeter = CalculateSize("perimeter");
        cv::Mat src = cv::imread("/home/user/Program/ls-dev/dispatchProject/UnitTest/fisheye/interface/imgs_ori/ori.dat");
        cv::Mat dst_center(size_center, CV_8UC3);
        cv::Mat dst_perimeter(size_perimeter, CV_8UC3);
        cv::Mat dst_perimeter_resize(cv::Size(size_perimeter.width / 2, size_perimeter.height / 2), CV_8UC3);

        cv::VideoCapture cap(FLAGS_input.c_str());

        if (!cap.isOpened()) {
            std::cerr << "Unable to open video file for capturing" << std::endl;
        }

        int fourcc = int(cap.get(cv::CAP_PROP_FOURCC));
        double fps = cap.get(cv::CAP_PROP_FPS);
        int frame_count = int(cap.get(cv::CAP_PROP_FRAME_COUNT));
        int current_frame = 0, width = 1280, height = 640;
//        cap.read(src);
//        cv::imwrite("fish_ori.jpg", src);

//    cv::VideoWriter writer(FLAGS_output.c_str(), CV_FOURCC('M', 'P', '4', 'V'), 20, size_perimeter);

        Mat resize_src;
//        resize(src, resize_src, cv::Size(src.cols, src.rows));

        cout << src.cols << " " << src.rows << " " << src.type() << endl;
        cout << dst_perimeter.cols << "  " << dst_perimeter.rows << endl;
        cout << dst_center.cols << "  " << dst_center.rows << endl;

        cameraGHandler* camera = new cameraGHandler(src.cols, src.rows, src.data,
                                                    dst_perimeter.cols, dst_perimeter.rows, dst_perimeter.data,
                                                    FLAGS_perimeter_top_angle,
                                                    FLAGS_perimeter_bottom_angle);

        int frame_num = 1000;
        int64_t sum = 0;
        while (cap.read(src)){
            if (frame_num > 1010) break;
            frame_num++;

            int64_t start = getCurrentTime();
            camera->run(src.data);
            int64_t end = getCurrentTime();
            sum += (end-start);
            cout <<"mode : 0 frame num : " << frame_num << " cost : " << end-start << endl;

            string img_path, img_path_center, img_path_ori;
            img_path = "./imgs/" + to_string(frame_num) + ".jpg";
            img_path_center = "./imgs_center/" + to_string(frame_num) + ".jpg";
            img_path_ori = "./imgs_ori/" + to_string(frame_num) + ".jpg";

            cv::imwrite(img_path, dst_perimeter);
        }
        cap.release();

    };

    thread thread_center(center_func);
    thread thread_perimeter(perimeter_func);

    thread_center.join();
    thread_perimeter.join();

}

int main(int argc, char *argv[]){

    //    google::ParseCommandLineFlags(&argc, &argv, true);
    cv::Size size_center = CalculateSize("center");
    cv::Size size_perimeter = CalculateSize("perimeter");
    cv::Mat src;
    cv::Mat dst_center(size_center, CV_8UC3);
    cv::Mat dst_perimeter(size_perimeter, CV_8UC3);
    cv::Mat dst_perimeter_resize(cv::Size(size_perimeter.width / 2, size_perimeter.height / 2), CV_8UC3);

    cv::VideoCapture cap(FLAGS_input.c_str());

    if (!cap.isOpened()) {
        std::cerr << "Unable to open video file for capturing" << std::endl;
    }

    int fourcc = int(cap.get(cv::CAP_PROP_FOURCC));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int frame_count = int(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int current_frame = 0, width = 1280, height = 640;
    cap.read(src);
//    src = cv::imread("/home/user/workspace/xxs/DPH_Server/build/imgs_detections/10100.jpg");


//    cv::VideoWriter writer(FLAGS_output.c_str(), CV_FOURCC('M', 'P', '4', 'V'), 20, size_perimeter);

//    Mat resize_src;
//    resize(src, resize_src, cv::Size(src.cols, src.rows));

    cout << src.cols << "  " << src.rows << endl;
    cout << dst_perimeter.cols << "  " << dst_perimeter.rows << endl;
    cout << dst_center.cols << "  " << dst_center.rows << endl;

//    cameraGHandler* camera;

//    0: gpu, 1:cpu
//    if (mode == 0) {
    cameraHandler* camera_per = new cameraHandler(src.cols, src.rows, src.data,
                                    dst_perimeter.cols, dst_perimeter.rows, dst_perimeter.data,
                                    FLAGS_perimeter_top_angle,
                                    FLAGS_perimeter_bottom_angle);
//    } else {
//    cameraHandler* camera_center = new cameraHandler(src.cols, src.rows, src.data,
//                                    dst_center.cols, dst_center.rows, dst_center.data,
//                                    FLAGS_center_zoom_angle);
//    }

    int frame_num = 1000;
    int64_t sum = 0;
    while (cap.read(src)){
        if (frame_num > 1010) break;
        frame_num++;

        int64_t start = getCurrentTime();
        camera_per->run(src.data);

//        std::vector<float> output_ = {198, 677, 332, 336, 761, 420, 591, 940};
//        float input_[8];

//        下面代码块注释掉可以跑
//        std::vector<float> output_ = {1908, 1093, 2069, 1493};
//        float input_[4];
//        int x, y;
//        std::cout << "GetOutputPolygonFromInputPolygon" << std::endl;
//        for (unsigned int i = 0; i < output_.size() / 2; i++) {
//            float pan, tilt;
//            camera_per->GetPositionFromOutputVideoPoint(int(output_[2 * i] ), int(output_[2 * i + 1]), &pan, &tilt);
//            cout << "pan : " << pan << " tilt : " << tilt <<endl;
//            camera_per->GetInputVideoPointFromPosition(pan, tilt, &x, &y);
//
////            camera_per->GetInputPointFromOutputPoint(int(output_[2 * i] ), int(output_[2 * i + 1] ), &x, &y);
//            input_[2 * i] = float(x), input_[2 * i + 1] = float(y);
////            std::cout << x * 0.4 << ',' << y * 0.4 << ':' << output_[i * 2] << ',' << output_[i * 2 + 1] << "  |  ";
//        }
//        cv::rectangle(dst_perimeter,cvPoint(output_[0] ,output_[1]),cvPoint(output_[2],output_[3]),Scalar(0,255,0),1,1,0);
//        cv::rectangle(src,cvPoint(input_[0] ,input_[1]),cvPoint(input_[2],input_[3]),Scalar(0,255,0),1,1,0);
//        std::cout << std::endl;

        float pan_0, tilt_0, pan_1, tilt_1;
        int input_x_0;
        int input_y_0;
        int input_x_1;
        int input_y_1;
        cout << "start transform" << endl;
        camera_per->GetPositionFromOutputVideoPoint(1908, 1093, &pan_0, &tilt_0);
        camera_per->GetInputVideoPointFromPosition(pan_0, tilt_0, &input_x_0, &input_y_0);
        cout << "x : " << input_x_0 << " y : " << input_y_0 <<endl;
//        cout << *input_x_0 << " " << *input_y_0 << endl;

        camera_per->GetPositionFromOutputVideoPoint(2069, 1493, &pan_1, &tilt_1);
        camera_per->GetInputVideoPointFromPosition(pan_1, tilt_1, &input_x_1, &input_y_1);
        cout << "x : " << input_x_1 << " y : " << input_y_1 <<endl;
        cout << "end transform" << endl;

//
//        cv::rectangle(dst_perimeter,cvPoint(output_[0] ,output_[1]),cvPoint(output_[2],output_[3]),Scalar(0,255,0),1,1,0);
        cv::rectangle(src,cvPoint(input_x_0 ,input_y_0),cvPoint(input_x_1,input_y_1),Scalar(0,255,0),1,1,0);




//        camera_center->run(src.data);
        int64_t end = getCurrentTime();
        sum += (end-start);
//        cout <<"mode : " << mode << " frame num : " << frame_num << " cost : " << end-start << endl;

        string img_path, img_path_center, img_path_ori;
        img_path = "./imgs/" + to_string(frame_num) + ".jpg";
        img_path_center = "./imgs_center/" + to_string(frame_num) + ".jpg";
        img_path_ori = "./imgs_ori/" + to_string(frame_num) + ".jpg";
//        if (mode == 0) {
            cv::imwrite(img_path, dst_perimeter);
            cv::imwrite(img_path_ori, src);
//        } else {
//            cv::imwrite(img_path_center, dst_center);
//        }


    }
    cout << "avg time cost : " << sum/100 << endl;

    return 0;
}

//export DISPLAY=:1