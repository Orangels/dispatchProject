//
// Created by 李大冲 on 2019-08-26.
//
#ifdef __JETBRAINS_IDE__
#define DEFINE_string
#define DEFINE_double
//how it works: https://stackoverflow.com/questions/39980645/enable-code-indexing-of-cuda-in-clion/46101718#46101718
#endif


#include "infer_.h"
#include "python_route.h"
#include <gflags/gflags.h>
#include <thread>
#include <exception>
#include <time.h>

#define sign(x) ( ((x) <0 )? -1 : ((x)> 0) )

DEFINE_string(ipt , "", "input video file name");
DEFINE_string(opt, "", "output video file name");
DEFINE_string(mac, "", "Mac Id");
DEFINE_string(yaml, "", "path to yaml");
DEFINE_double(media_id, 0.0, "media_id");


unsigned long GetTickCount() {
    // https://blog.csdn.net/guang11cheng/article/details/6865992
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

float GetTickCount2() {
    std::clock_t clock = std::clock();
    return 1000.0 * clock / CLOCKS_PER_SEC;
}

bool loadKeyParams(std::string yaml, int &s) {
//    std::cout << "load----" << std::endl;
    cv::FileStorage fs(yaml, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        fprintf(stderr, "%s:%d:loadParams falied. 'yaml' does not exist\n", __FILE__, __LINE__);
        return false;
    }
    std::string data, pic;
    fs["stop"] >> data;
    fs["pic"] >> pic;
    fs.release();
    s = 0;
//    std::cout << "read p----" << std::endl;
    if (pic == "y") {
        s = 1;
        fs.open(yaml, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            fprintf(stderr, "%s:%d:loadParams falied. 'yaml' does not exist\n", __FILE__, __LINE__);
            return false;
        }
        fs << "stop" << data;
        fs << "pic" << "n";
        fs.release();
        //return true;
    }
    if (data == "y") return true;
    data[0] -= 32;
    if (data == "Y")return true;
    return false;
}

Infer_RT::~Infer_RT() {
    if (engine)delete (engine);
    delete[] scores;
    delete[] boxes;
    delete[] classes;

}

Infer_RT::Infer_RT(const char *engine_file, const char *input,
                   int meida_id, const char *mac, const char *cyaml,
                   int device, std::string modes) :
        meida_id(meida_id), mac(mac), stop(false), baseOnSrc(false), yaml(cyaml) {
    if (cyaml == NULL) {
        std::cerr << "yaml should be set!" << std::endl;
        throw "error";
    }
    line = yaml;
    line = line.replace(line.find("yaml"), 4, "jpg");
    string fcos = "fcos", ef = engine_file;
    if (ef.find(fcos) > ef.length()) cout << "Using retinanet..." << endl;
    else cout << "Using Fcos..." << endl;
    printf("engine_file: %s\ninput: %s\nmac: %s media id:%d\n",
           engine_file, input, mac, meida_id);
    engine = new def_retinanet::Engine(engine_file);
    std::cout <<"build engin ok"<< std::endl;
    setInfo(engine, input, device, modes);
    printf("engine Init location:%p...\n", engine);
}

template<typename T>
void Infer_RT::setInfo(T *ptr, const char *input, int device, std::string modes) {
    printf("mac2: %s media id:%d\n", mac, meida_id);
    std::cout << "p: param_" + to_string(meida_id) + ".yaml" << std::endl;
    auto inputSize = ptr->getInputSize();
    num_det = ptr->getMaxDetections();
    auto batch = ptr->getMaxBatchSize();

    channels = 3;
    run_batch = 3;
    slots = 2;
    height = inputSize[0];
    width = inputSize[1];
    std::cout << "inputSize:" << height << ',' << width << "num_det:" << num_det << "batch:" << batch << std::endl;

    bool saveVideo = true;//false;//
    bool savePhoto = false;//true;// 
    //目前来看，没必要设置多个slots，因为他们内部是完全同步的，没有异步的操作
    dewarper = new deWarp(width, height, slots + 1, modes, saveVideo, savePhoto);
    dewarper->readVideo(input, device);
    h_ratio = dewarper->ratio_h;
    w_ratio = dewarper->ratio_w;
    std::cout << "h_ratio:" << h_ratio << " w_ratio: " << w_ratio <<
              "|h:" << height << " w:" << width << std::endl;
    // Create device buffers
    cudaMalloc(&scores_d, batch * num_det * sizeof(float));
    cudaMalloc(&boxes_d, batch * num_det * 4 * sizeof(float));
    cudaMalloc(&classes_d, batch * num_det * sizeof(float));

    int useful_batch = run_batch * slots;
    scores = new float[useful_batch * num_det];
    boxes = new float[useful_batch * num_det * 4];
    classes = new float[useful_batch * num_det];
    N = 1, n_post = n_count = slots - 1;
    N_s = run_batch * num_det;
    N_b = run_batch * num_det * 4;
    show_ratio_h = show_ratio_w = 0.4;
    if (baseOnSrc) ResImgSiz = cv::Size(dewarper->cols * show_ratio_w, dewarper->rows * show_ratio_h);
    else ResImgSiz = cv::Size(dewarper->col_out * show_ratio_w, dewarper->row_out * show_ratio_h);
    std::cout << "output size:" << ResImgSiz << std::endl;
    for (int iii = 0; iii < 2; iii++) showDsts.push_back(cv::Mat(ResImgSiz, CV_8UC3));
    //python interpreter
    std::cout << "python now!" << std::endl;
    pr = new python_route(h_ratio, w_ratio, dewarper->row_out, dewarper->col_out);
    std::cout << "python over!" << std::endl;
    testStar = true;
}


void Infer_RT::getsrc() {
    /*用来将展开切分结果输出，用python检查结果，在异步时输出尺寸会受限制（原因暂不明）*/
    int sz = width * height * channels * run_batch;
    float *data = new float[sz];
    cudaMemcpy(data, dewarper->data, sz * sizeof(float), cudaMemcpyDeviceToHost);
    pr->PythonInfer(run_batch, height, width, data);
    delete[](data);
}

void Infer_RT::preprocess(bool joinAll, bool cpdst) {
    //std::cout << "joinAll" << joinAll << "cpdst" << cpdst << std::endl;
    unsigned long beg = GetTickCount();
    if (joinAll)dewarper->join_thread();
    dewarper->process(cpdst, baseOnSrc);
    unsigned long end = GetTickCount();
    std::cout << "preProcess: " << end - beg << "ms\n";

}

void Infer_RT::process_() {
    unsigned long beg = GetTickCount();
    dewarper->currentStatus();

    //std::cout << "Running inference..." << std::endl;
    vector<void *> buffers = {dewarper->data, scores_d, boxes_d, classes_d};
    engine->infer(buffers, run_batch);
    // Get back the bounding boxes
    cudaMemcpy(scores + n_count * N_s, scores_d, sizeof(float) * num_det * run_batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(boxes + n_count * N_b, boxes_d, sizeof(float) * num_det * 4 * run_batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(classes + n_count * N_s, classes_d, sizeof(float) * num_det * run_batch, cudaMemcpyDeviceToHost);

    std::cout << "kernel infer: " << GetTickCount() - beg << "ms\n";
}


//void Infer_RT::cal_src_ploygon(cv::Mat src, cv::Mat dst) {
//    /*点映射回原图点方法*/
//    int input_X[] = {100, 700, 700, 100}, input_Y[] = {200, 200, 900, 900}, nn_points = 4;
//    int n_points = 0, total_N = 100;
//    int *ret;// = expand_line(input_X, input_Y, nn_points, n_points, total_N);
//
//    int output_x[n_points], output_y[n_points];
//    int *input_x = ret, *input_y = ret + total_N;
//    dewarper->mappingPolygon(n_points, input_x, input_y, output_x, output_y);
//
//    for (int i = 0; i < n_points; i++) {
//        cv::line(src, cv::Point(output_x[(i + 1) % n_points], output_y[(i + 1) % n_points]),
//                 cv::Point(output_x[i], output_y[i]), cv::Scalar(50, 50, 50), 3);
//        cv::line(dst, cv::Point(input_x[(i + 1) % n_points], input_y[(i + 1) % n_points]),
//                 cv::Point(input_x[i], input_y[i]), cv::Scalar(50, 50, 50), 3);
//    }
//    delete[] ret;
//    cv::imwrite("src.jpg", src);
//    cv::imwrite("dst.jpg", dst);
//}

void Infer_RT::_getdst(int No, int eclipse) {
    unsigned long beg = GetTickCount();
    //用来传输、保存等，非高频需求

    cv::Mat dst = showDsts[dst_curid];
    //先判断是否有数据，否则就用错位的数据吧
    if (dewarper->dst.empty() or dst.empty()) {
        //https://blog.csdn.net/jueshiwushuang2007/article/details/8855665
        std::cout << "dst data is empty!";
        return;
    }
    if (baseOnSrc) cv::resize(dewarper->src_clone, dst, ResImgSiz);
    else cv::resize(dewarper->dst, dst, ResImgSiz);
    dewarper->currentImg();
    cv::putText(dst, "Frame No: " + std::to_string(No) + ". Eclipse: " + std::to_string(eclipse) +
                     "ms.", cv::Point(int(1.2 * width * show_ratio_w), int(0.2 * height * show_ratio_h)),
                cv::FONT_HERSHEY_PLAIN, 5.0 * show_ratio_h, cv::Scalar(255, 245, 250), int(3 * show_ratio_h));
    unsigned long end = GetTickCount();
    std::cout << "get_dst: " << end - beg << "ms\n";
}

void Infer_RT::decorate() {
    unsigned long beg = GetTickCount();
    //严格要求数据非同步读写，box和dst
    send_cur = cur;
    cur = dst_curid;
    dst_curid = (dst_curid + 1) % 2;
    if (showDsts[cur].empty()) {
        //https://blog.csdn.net/jueshiwushuang2007/article/details/8855665
        std::cout << "decorate data is empty!" << "postProcess cur: " << showDsts[cur].empty() << ','
                  << showDsts[cur].rows << ',' << cur << ',' << dst_curid << "\n";
        return;
    }
    auto *dobj = baseOnSrc ? dewarper : nullptr;
    pr->ParseRet(showDsts[cur], show_ratio_w, show_ratio_h, dobj);
    dewarper->saveImg(line, showDsts[cur]);// vedio or img
//    std::cout << 'r' << showDsts[cur].rows << 'w' << showDsts[cur].cols << std::endl;
    unsigned long end = GetTickCount();
    std::cout << "decorate: " << end - beg << "ms\n";
}

void Infer_RT::postprocess(int current_frame_id, bool kept, bool &tosend) {
    unsigned long beg = GetTickCount();
    //std::cout << "tosend: " << tosend << "k: " << kept << "send cur: " << send_cur << "cur: " << cur << std::endl;
    n_post = n_count;
    n_count = (n_count + 1) % slots;
    int base = n_post * N_s;
    cv::Mat Res;
    if (tosend) {
        Res = showDsts[send_cur];
        tosend = false;
    }
    //不能两个python调用同时进行，会导致[Python] Fatal error: GC object already tracked问题！
    pr->PythonPost(&boxes[base * 4], &scores[base], &classes[base], run_batch, num_det, kept,
                   Res, meida_id, current_frame_id, mac);
    unsigned long end = GetTickCount();
    std::cout << "postProcess: " << end - beg << "ms\n";
}

void Infer_RT::_fowShow() {
    unsigned long beg = GetTickCount();
    n_post = (n_post + 1) % slots;
    dewarper->currentImg();
    int base = n_post * N_s, i = 0, h = 0, w = 0;
    int row = dewarper->dst.rows, col = dewarper->dst.cols;
    int half_h = row / 2, qurty_w = col / 3;

    for (int j = 0; j < run_batch * num_det; j++) {
        if (j / num_det == 1)continue;
        if (j / num_det == 2) {
            h = half_h;
            w = qurty_w;
        } else {
            h = w = 0;
        }
        i = j + base;
        // Show results over confidence threshold
        if (scores[i] >= 0.3) {//and classes[i] == 0
            float x1 = boxes[i * 4 + 0] * w_ratio + w;
            float y1 = boxes[i * 4 + 1] * h_ratio + h;
            float x2 = boxes[i * 4 + 2] * w_ratio + w;
            float y2 = boxes[i * 4 + 3] * h_ratio + h;
            cout << "{" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "} ";
            //scores[i] = classes[i] = 0;
            //for (int k = 0; k < 4; k++)boxes[i * 4 + k] = 0;
            //cout << "Found box {";
            //for (int k = 0; k < 4; k++) cout << boxes[i * 4 + k] << ", ";
            //cout << "}\n";
            // Draw bounding box on image
            cv::rectangle(dewarper->dst, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0));
        }
    }
    string res = "detections" + to_string(N++) + ".jpg";
    std::cout << "save img to:" << res << std::endl;
    // Write image
    cv::Mat ResImg;
    dewarper->saveImg(res, ResImg);
    unsigned long end = GetTickCount();
    std::cout << "\n postProcessForshow: " << end - beg << "ms\n";
}

//about thread: 1. https://www.cnblogs.com/wangguchangqing/p/6134635.html
//              2. https://blog.csdn.net/hai008007/article/details/80246437
void Infer_RT::run() {
    unsigned long beg = GetTickCount(), t1;
    std::thread infer_, tail, isStop, rzdst, decor;
    int i = 0, eclipse = 0, carrayOnGap = 3;
    bool tocanvas = false, stop, baseImg = false, dst_t = false, sendDecorate = false;

    auto fishes = [=, &stop, &i, &baseImg]() {
        unsigned long beg = GetTickCount();
        int saveImg = 0;
        stop = loadKeyParams(yaml, saveImg);
        if (saveImg != 0) baseImg = true;
        //std::cout << "join dst:" << i << ',' << (i % carrayOnGap == 0) << std::endl;
        if (i > 0 && i % carrayOnGap == 0) {
            dewarper->join_dst();//需在_getdst之前完成
            if (baseImg) {
                cv::Mat ResImg = baseOnSrc ? dewarper->src_clone : dewarper->dst;
                dewarper->saveImg(line, ResImg, baseImg);
                baseImg = false;
            }
        }
        //std::cout << "join dst2:" << std::endl;
        unsigned long end = GetTickCount();
        std::cout << "fishes: " << end - beg << "ms" << std::endl;
    };
    preprocess();
    while (true) {
        t1 = GetTickCount();
        // 线程1：检查是否需要停止循环；是否保存底图
        isStop = std::thread(fishes);//当前循环完成2ms
        // 线程2：获取展开图
        if (i > 0 && i % carrayOnGap == 0) {
            rzdst = std::thread([=] { _getdst(i, eclipse); });//resize dst||当前循环完成5ms
            dst_t = true;
            if (tocanvas) {//不用太频繁的刷新
                //std::cout << "end decorate:" << i << std::endl;
                tocanvas = false;
                decor.join();
                sendDecorate = true;
            }
        }
        // 线程3：模型运行
        //关于输出的box、score等||在infer之前确定写入那个，空出哪个槽位
        infer_ = std::thread([=] { process_(); });//model 当前循环完成54ms

        preprocess(true, (i + 1) % carrayOnGap == 0);//read img 当前循环完成50-60ms
        isStop.join();//2ms
        infer_.join();//54ms

        // 线程4：跟踪、发送
        if (i > 0)tail.join();
        tail = std::thread([=, &sendDecorate] { postprocess(i - 1, (i + 1) % carrayOnGap == 0, sendDecorate); });

        // 线程5：画线等
        if (i > 0 && i % carrayOnGap == 0) {//时间差不多是比前传2倍多一点
            rzdst.join();//在decorate之前结束 5ms
            tocanvas = true;
            dst_t = false;
            decor = std::thread([=] { decorate(); });
        }

        eclipse = int(GetTickCount() - t1);
        std::cout << n_post << n_count << "--- No." << ++i << " & " << eclipse <<
                  " ms ---" << !dewarper->has_frame << " s:" << stop << std::endl << std::endl;
        if (!dewarper->has_frame or stop)break;
    }
    if (stop) std::cout << "stop by user!" << std::endl;
    else std::cout << "stop by No frames!" << std::endl;
    if (dst_t)rzdst.join();
    if (tocanvas)decor.join();
    tail.join();
    unsigned long end = GetTickCount();
    std::cout << "Mean time: " << 1.0 * (end - beg) / i << "ms " << std::endl;
}

int main(int argc, char *argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_ipt.empty()) {
        std::cerr << "Invalid input video file name" << std::endl;
        return -1;
    }
    if (FLAGS_opt.empty()) {
        std::cerr << "Invalid output video file name" << std::endl;
        return -1;
    }
    std::cout << "hree 1"<< std::endl;
//    Infer_RT inferss = Infer_RT("/home/user/weight/int8_640_1280.plan", FLAGS_ipt.c_str());
    Infer_RT inferss = Infer_RT("/home/user/project/fisheye/weights/fcos_f16_640_1280.plan",
                                FLAGS_ipt.c_str(), int(FLAGS_media_id), FLAGS_mac.c_str(), FLAGS_yaml.c_str());
    std::cout << "hree 2"<< std::endl;
    /*
    auto mode_order = [&]() {
        int N = 5;
        unsigned long beg = GetTickCount();
        for (int i = 0; i < N; i++) {
            std::cout << std::endl << "----------------------------" << std::endl;
            inferss.preprocess();
            inferss.process_();
            inferss.postprocess(i, 0, 0);
        }
        unsigned long end = GetTickCount();
        std::cout << "All time: " << 1.0 * (end - beg) / N << "ms " << std::endl;
    };
     */
    //inferss.run();
    #include <unistd.h>
    int i=0, carrayOnGap = 3;
    inferss.preprocess();
    while(true){
	    if (i > 0 && i % carrayOnGap == 0) {
            inferss.dewarper->join_dst();}
            inferss.dewarper->currentStatus();
	    inferss.preprocess(true, (i++ + 1) % carrayOnGap == 0);
	    if (!inferss.dewarper->has_frame)break;
	    sleep(2);
    }
//    mode_order();
    return 0;
}

// export DISPLAY=:0.0
// ./infer_ --ipt s.mp4 --opt 'a' --media_id 1 --mac '00-02-D1' --yaml param_1.yaml
