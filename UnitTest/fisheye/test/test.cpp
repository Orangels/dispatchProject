#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <gflags/gflags.h>
#include "IMV1.h"

DEFINE_string(lens, "vivotek", "camera lens");
//DEFINE_string(mode, "center", "dewarping mode");
DEFINE_string(mode, "perimeter", "dewarping mode");
DEFINE_string(input, "rtsp://root:admin123@192.168.88.67/live.sdp", "input video file name");
//DEFINE_string(input, "rtsp://root:admin123@192.168.88.26/live.sdp", "input video file name");
DEFINE_string(output, "perimeter_ls_test.mp4", "output video file name");

DEFINE_double(camera_rotation, 0.0, "camera rotation degree");
DEFINE_double(pixels_per_degree, 16.0, "pixels per degree");
DEFINE_double(center_zoom_angle, 90.0, "center zoom field of view");
DEFINE_double(perimeter_top_angle, 90.0, "perimeter top angle");
DEFINE_double(perimeter_bottom_angle, 30.0, "perimeter bottom angle");

using namespace std;
using namespace cv;

cv::Size CalculateSize() {
    if (FLAGS_mode == "center") {
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
    cv::Size size = CalculateSize();
    cv::Mat src;
    cv::Mat dst(size, CV_8UC3);
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
    cout << src.cols << "  " << src.rows << endl;
    
//    const char *rpl = FLAGS_lens == "vivotek" ? "B9VVT" : "C9VVT";
    const char *rpl = "B9VVT";

    IMV_Buffer *inBuffer, *outBuffer;

    IMV_CameraInterface *camera = new IMV_CameraInterface();

    inBuffer = new IMV_Buffer;

    outBuffer = new IMV_Buffer;

    inBuffer->width = src.cols;

    inBuffer->height = src.rows;

    inBuffer->frameX = 0;

    inBuffer->frameY = 0;

    inBuffer->frameWidth = src.cols;

    inBuffer->frameHeight = src.rows;

    inBuffer->data = src.data; //videoFrameData is a pointer

    outBuffer->width = dst.cols;

    outBuffer->height = dst.rows;

    outBuffer->frameX = 0;

    outBuffer->frameY = 0;

    outBuffer->frameWidth = dst.cols;

    outBuffer->frameHeight = dst.rows;

    outBuffer->data = dst.data ; //displayData is the pointer to

    camera->SetLens(strdup(rpl)); //we indicates the lens type we use

    camera->SetFiltering(IMV_Defs::E_FILTER_BILINEAR);

    unsigned long iResult = camera->SetVideoParams(

            inBuffer,

            outBuffer,

            IMV_Defs::E_BGR_24_STD | IMV_Defs::E_OBUF_TOPBOTTOM,

            IMV_Defs::E_VTYPE_PERI,
//            IMV_Defs::E_VTYPE_PERI_CUSTOM,
//            IMV_Defs::E_VTYPE_QUAD,

            IMV_Defs::E_CPOS_CEILING);

    cout << iResult << endl;
    if (iResult == 0)

    {

        //the library is correctly initialized
        cout << "IMV Init suc" << endl;
    }

    else

    {

        //an error occurred
        cout << "IMV Init error" << endl;

    }

    camera->SetZoomLimits(40.f,180.f);
    camera->SetTiltLimits(FLAGS_perimeter_top_angle - 90.0f, FLAGS_perimeter_bottom_angle - 90.0f);
    camera->Update();
    imwrite("imgs/test_1.jpg", dst);
    int frame_num = 1000;
    while (cap.read(src)){
        if (frame_num > 1100) break;
        frame_num++;
        cout << "frame num : " << frame_num << endl;
        inBuffer->data = src.data;
        camera->Update();
        string img_path, img_path_ori;
        img_path = "./imgs/" + to_string(frame_num) + ".jpg";
        cv::imwrite(img_path, dst);
    }



    return 0;
}