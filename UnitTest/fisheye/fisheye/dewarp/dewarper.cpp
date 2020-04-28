#ifdef __JETBRAINS_IDE__
#define DEFINE_string
#define DEFINE_double
//how it works: https://stackoverflow.com/questions/39980645/enable-code-indexing-of-cuda-in-clion/46101718#46101718
#endif

//#include <iostream>
//#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "camera_view.h"
#include "dewarper.h"
//#include <pthread.h>
#include <unistd.h>
#include <future>

std::unordered_map<std::string, CameraView *> CameraViewInstance;

//pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;/*初始化互斥锁*/
//pthread_cond_t cond = PTHREAD_COND_INITIALIZER;//init cond

DEFINE_string(lens, "vivotek", "camera lens");
DEFINE_string(mode, "center", "dewarping mode");
DEFINE_string(input, "", "input video file name");
DEFINE_string(output, "", "output video file name");

DEFINE_double(camera_rotation, 0.0, "camera rotation degree");
DEFINE_double(pixels_per_degree, 16.0, "pixels per degree");
DEFINE_double(center_zoom_angle, 90.0, "center zoom field of view");
DEFINE_double(perimeter_top_angle, 90.0, "perimeter top angle");
DEFINE_double(perimeter_bottom_angle, 30.0, "perimeter bottom angle");

bool InitGLContext() {
    if (glfwInit() != GLFW_TRUE) {
        return false;
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow *window = glfwCreateWindow(640, 480, "", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return false;
    }
    return true;
}

void DestroyGLContext() {
    glfwDestroyWindow(glfwGetCurrentContext());
    glfwTerminate();
}


/*
extern "C" {
int run(uchar *input, int row, int col,
        uchar *output, int row_out, int col_out,
        int current_frame,
        const char *lens, const char *mode,
        int camera_rotation = 0,
        int center_zoom_angle = 90,
        int perimeter_top_angle = 95,
        int perimeter_bottom_angle = 30) {

    CameraView *&camera = CameraViewInstance["camera"];

    FLAGS_mode = mode;
    FLAGS_lens = lens;
    FLAGS_camera_rotation = float(camera_rotation);
    FLAGS_perimeter_top_angle = float(perimeter_top_angle);
    FLAGS_perimeter_bottom_angle = float(perimeter_bottom_angle);
    FLAGS_center_zoom_angle = float(FLAGS_center_zoom_angle);
    clock_t time;
    bool show_time = false;

    if (current_frame == 1) {
        if (!InitGLContext()) {
            std::cerr << "Unable to initialize OpenGL context" << std::endl;
            return -1;
        }
        const char *rpl = FLAGS_lens == "vivotek" ? "B9VVT" : "C9VVT";
        if (FLAGS_mode == "center") {
            camera = new CameraView(col, row, input,
                                    col_out, row_out, output, rpl,
                                    FLAGS_center_zoom_angle);
        } else {
            camera = new CameraView(col, row, input,
                                    col_out, row_out, output, rpl,
                                    FLAGS_perimeter_top_angle,
                                    FLAGS_perimeter_bottom_angle);
        }
    }
    camera->input_buffer.data = input;

    camera->Process();
    return 1;
}
}
*/


deWarp::deWarp(int w, int h, int slot, std::string modes, bool saveVideo, bool savePhoto,
               std::string len, float camera_rotation, float pixels_per_degree,
               float center_zoom_angle, float perimeter_top_angle,
               float perimeter_bottom_angle) :
        width(w), height(h), slots(slot), lens(len), mode(modes), save_video(saveVideo), save_photo(savePhoto),
        current_frame(0), camera_rotation(camera_rotation), pixels_per_degree(pixels_per_degree),
        center_zoom_angle(center_zoom_angle), perimeter_top_angle(perimeter_top_angle),
        perimeter_bottom_angle(perimeter_bottom_angle) {
    if (lens != "vivotek" && lens != "3s") {
        std::cerr << "Invalid camera lens: " << std::endl;
        throw "error";
    }
    if (mode != "center" && mode != "perimeter") {
        std::cerr << "Invalid dewarping mode" << std::endl;
        throw "error";
    }
    rpl = (lens == "vivotek") ? "B9VVT" : "C9VVT";
    if (mode == "center") {
        int length = int(pixels_per_degree * center_zoom_angle / 4) << 2;
        row_out = col_out = length;
    } else {
        col_out = int(pixels_per_degree * 180.0f / 4) << 2;
        row_out = int(pixels_per_degree * (perimeter_top_angle - perimeter_bottom_angle) / 2) << 2;
    }
    has_frame = true, init_ = false;
    slots = slots > 0 ? slots : 1;//最小为1
    //感觉这儿逻辑设计过于复杂
//    n_slots = last_slots = slots - 1;
    dst_slots = 0;
}

deWarp::~deWarp() {
    delete (((CameraView *) camera)->sp);
    delete ((CameraView *) camera);
    DestroyGLContext();
    cv::destroyWindow("Display window");
    if (save_video) writer.release();
    cap.release();
}

void deWarp::readVideo(const char *input, int device) {
    //https://blog.csdn.net/caomin1hao/article/details/83057587
    if (!InitGLContext()) {
        std::cerr << "Unable to initialize OpenGL context" << std::endl;
        throw "error";
    }
    if (input != NULL)
        cap.open(input);
    else if (device != -1)
        cap.open(device);
    else {
        std::cerr << "you need input file name of a vedio or camera device No." << std::endl;
        throw "error";
    }

    if (!cap.isOpened()) {
        std::cerr << "Unable to open video file for capturing" << std::endl;
        throw "error";
    }

    fourcc = int(cap.get(cv::CAP_PROP_FOURCC));
    fps = cap.get(cv::CAP_PROP_FPS);
    format = cap.get(cv::CAP_PROP_FORMAT);
    frame_count = int(cap.get(cv::CAP_PROP_FRAME_COUNT));
    rows = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cols = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    extraDst = 0;
    for (int i = 0; i < slots + extraDst; i++) {
        status.push_back(false);
        dsts.push_back(cv::Mat(cv::Size(col_out, row_out), CV_8UC3));
    }

    dst = dsts[0];
    cap.read(src);
    camera = new CameraView(cols, rows, src.data,
                            col_out, row_out, dst.data,
                            rpl, perimeter_top_angle,
                            perimeter_bottom_angle);
    ((CameraView *) camera)->sp = new splitProcess(col_out, row_out, slots, width, height, slots);
    ratio_w = ((CameraView *) camera)->sp->ratio_width;
    ratio_h = ((CameraView *) camera)->sp->ratio_height;
    for (int iii = 0; iii < 2; iii++) {
        //for check empty
        cv::Mat tempM;
        cap >> tempM;
        mul_mat.push(tempM);
    }
    currentImg();
}

void deWarp::saveImg(std::string filename, cv::Mat ResImg, bool savebase) {
    cv::Size ResImgSiz;
    if (ResImg.data) {
        ResImgSiz = cv::Size(ResImg.cols, ResImg.rows);
    } else if ((!save_photo and !save_video) or save_video) {
        ResImgSiz = cv::Size(col_out * 0.5, row_out * 0.5);
        ResImg = cv::Mat(ResImgSiz, CV_8UC3);
        cv::resize(dst, ResImg, ResImgSiz);
    }
    std::string rs;
    if (!init_) {
        char randomX[] = {"definerandom"};
        int randomN = rand() % (sizeof(randomX));
        rs = &(randomX[randomN]);
    }
    if (savebase) {
        std::cout << "save base photo now!" << std::endl;
        ResImgSiz = cv::Size(col_out * 0.4, row_out * 0.4);
        cv::Mat tmpImg;
        cv::resize(ResImg, tmpImg, ResImgSiz);
        cv::imwrite(filename, tmpImg);
    } else if (save_video) {
        if (!init_) {
            init_ = true;
            std::cout << "Init vedio: " << ResImgSiz << "col:" << ResImg.cols << " row" << ResImg.rows << std::endl;
            //    writer.open("dete_" + s, fourcc, fps, cv::Size(col_out, row_out));
            writer.open("dete_" + rs + "_.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 8, ResImgSiz);
            if (!writer.isOpened()) {
                std::cerr << "Unable to open video file for writing" << std::endl;
                throw "error";
            }
        }
        writer.write(ResImg);
    } else if (save_photo) {
        //std::cout << "save photo now!" << std::endl;
        cv::imwrite(filename, ResImg);
    } else {
        if (!init_) {
            init_ = true;
            cv::namedWindow("Display window" + rs, cv::WINDOW_AUTOSIZE);
        }
        cv::imshow("Display window", ResImg);
        cv::waitKey(5);
    }
}

bool deWarp::checkStatus() {
    for (int i = 0; i < status.size(); i++) {
        if (!status[i])return true;
    }
    return false;
}

void deWarp::process(bool cpdst, bool baseOnSrc) {
    readSwitch = false;
    /* 1.read img to queqe异步读取*/
    readsrc = std::thread([=] {
        //尽可能多的读入队列
        do {
            cv::Mat tem;
            // 读取当前帧： https://www.cnblogs.com/little-monkey/p/7162340.html
            if (cap.read(tem))
                mul_mat.push(tem);
        } while (!readSwitch and mul_mat.size() < 20);
    });

    /* 2.run main function展开及切图成块*/
    if (cpdst)
        getdst = std::thread([=] {
            if (baseOnSrc)src.copyTo(src_clone);
            else ((CameraView *) camera)->getOutput();
        });

    mul_mat.front().copyTo(src);
    mul_mat.pop();
    ++current_frame;

    /*3. 获取底图*/
    ((CameraView *) camera)->Process();

    /*4. 跳帧*/
    for (int ii = 0; ii < mul_mat.size() / 6; ii++)mul_mat.pop();
    for (int i = 0; i < 0 and mul_mat.size() >= 6; i++) {
        std::cout << "pop:" << mul_mat.size();
        mul_mat.pop();
    }
    /*5. 退出条件*/
    if (mul_mat.empty())has_frame = false;//队列空，则退出程序
    std::cout << "source frames:" << mul_mat.size() << ',' << has_frame << std::endl;
}

void deWarp::join_thread() {
    readSwitch = true;
    readsrc.join();//more time to stop
}

void deWarp::join_dst() {
    getdst.join();
}

void deWarp::currentImg() {
    dst_slots = (dst_slots + 1) % (slots + extraDst);
    dst = dsts[dst_slots];
    ((CameraView *) camera)->resetOutput(dst.data);
}

void deWarp::currentStatus() {
    ((CameraView *) camera)->sp->get_refer();
    data = ((CameraView *) camera)->sp->refer_dyn;
}

void deWarp::mappingPolygon(std::vector<float> &output_, float *input_) {
    ((CameraView *) camera)->GetInputPolygonFromOutputPolygon(output_, input_);
}

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


int main_test(int argc, char *argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_lens != "vivotek" && FLAGS_lens != "3s") {
        std::cerr << "Invalid camera lens" << std::endl;
        return -1;
    }
    if (FLAGS_mode != "center" && FLAGS_mode != "perimeter") {
        std::cerr << "Invalid dewarping mode" << std::endl;
        return -1;
    }
    if (FLAGS_input.empty()) {
        std::cerr << "Invalid input video file name" << std::endl;
        return -1;
    }
    if (FLAGS_output.empty()) {
        std::cerr << "Invalid output video file name" << std::endl;
        return -1;
    }

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

    cv::VideoWriter writer(FLAGS_output.c_str(), fourcc, fps, size);
    if (!writer.isOpened()) {
        std::cerr << "Unable to open video file for writing" << std::endl;
        return -1;
    }

    if (!InitGLContext()) {
        std::cerr << "Unable to initialize OpenGL context" << std::endl;
        return -1;
    }

    CameraView *camera;
    const char *rpl = FLAGS_lens == "vivotek" ? "B9VVT" : "C9VVT";

    // TODO: Turns out src.data doesn't change
    while (cap.read(src)) {
        if (current_frame > 10)break;
        ++current_frame;
        std::cout << "Processing frame " << current_frame
                  << " of " << frame_count << "...\t";
        std::cout.flush();

        if (current_frame == 1) {
            if (FLAGS_mode == "center") {
                camera = new CameraView(src.cols, src.rows, src.data,
                                        dst.cols, dst.rows, dst.data, rpl,
                                        FLAGS_center_zoom_angle);
            } else {
                camera = new CameraView(src.cols, src.rows, src.data,
                                        dst.cols, dst.rows, dst.data, rpl,
                                        FLAGS_perimeter_top_angle,
                                        FLAGS_perimeter_bottom_angle);
            }
            ((CameraView *) camera)->sp = new splitProcess(dst.cols, dst.rows, 3, width, height);

        }
        camera->Process();
        writer.write(dst);
    }
    std::cout << "end!" << std::endl;
    delete ((CameraView *) camera)->sp;
    delete camera;
    DestroyGLContext();
    writer.release();
    cap.release();

    return 0;
}
