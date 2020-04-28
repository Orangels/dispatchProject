#include <iostream>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>
#include "../csrc/engine.h"
#include <unordered_map>

using namespace std;
using namespace cv;
unordered_map<string, def_retinanet::Engine *> engineInstance;
unordered_map<string, void *> dataContianer;
unordered_map<string, vector<float>> inputContianer;

int data_resize_cpu(const char *img_file, vector<float> &data, int w, int h, int channels) {
    auto image = imread(img_file, IMREAD_COLOR);
    cv::resize(image, image, Size(w, h));
    cv::Mat pixels;
    image.convertTo(pixels, CV_32FC3, 1.0, 0);

    vector<float> img;
    if (pixels.isContinuous())
        img.assign((float *) pixels.datastart, (float *) pixels.dataend);
    else {
        cerr << "Error reading image " << img_file << endl;
        return -1;
    }

    vector<float> mean{102.9801, 115.9465, 122.7717};
    vector<float> std{1.0, 1.0, 1.0};

    for (int c = 0; c < channels; c++) {
        for (int j = 0, hw = w * h; j < hw; j++) {
            data[c * hw + j] = (img[channels * j + c] - mean[c]) / std[c];
        }
    }
    return 1;
}

extern "C" {
void initialize(const char *engine_file, char *name, int info[]) {
    string names(name);
    auto *&engine = engineInstance[names];
    cout << "here" << endl;
    engine = new def_retinanet::Engine(engine_file);
    printf("engine & %p", engine);
    auto inputSize = engine->getInputSize();
    auto num_det = engine->getMaxDetections();
    auto batch = engine->getMaxBatchSize();
    int channels = 3;
    //cout << "name type:" << typeid(names).name() <<
    //     "name type:" << typeid(*engine).name() <<
    //     "input size:" << inputSize[0] << "*" << inputSize[1]
    //     << "Numdet:" << num_det <<
    //     "batch:" << batch << endl;
    // Create device buffers
    void *data_d, *scores_d, *boxes_d, *classes_d;
    cudaMalloc(&data_d, batch * channels * inputSize[0] * inputSize[1] * sizeof(float));
    cudaMalloc(&scores_d, batch * num_det * sizeof(float));
    cudaMalloc(&boxes_d, batch * num_det * 4 * sizeof(float));
    cudaMalloc(&classes_d, batch * num_det * sizeof(float));
    dataContianer.insert(std::pair<string, void *>{"data", data_d});
    dataContianer.insert(std::pair<string, void *>{"score", scores_d});
    dataContianer.insert(std::pair<string, void *>{"box", boxes_d});
    dataContianer.insert(std::pair<string, void *>{"classes", classes_d});

    // input container
    vector<float> data(batch * channels * inputSize[0] * inputSize[1]);
    inputContianer.insert(std::pair<string, vector<float> >{"input", data});
    info[0] = int(inputSize[0]);
    info[1] = int(inputSize[1]);
    info[2] = int(batch);
    info[3] = int(num_det);
}

void process(float *img_input, float *scores, float *boxes, float *classes,
             int batch, const char *name) {
    //cout << "Loading engine..." << endl;
    string s(name);
    auto *&engine = engineInstance[s];
    //printf("engine & %p", engine);
    auto inputSize = engine->getInputSize();
    //cout << "Preparing data..." << endl;
    int channels = 3;
    vector<float> data = inputContianer["input"];
    memcpy(data.data(), (float *) img_input, batch * channels * inputSize[0] * inputSize[1] * sizeof(float));
    //for (auto j = 0; j < batch; j++) {
    //    for (auto i = 0; i < 5; i++) {
    //        auto pre = j * channels * inputSize[0] * inputSize[1];
    //        cout << i << " " << data[pre + i] << ";";
    //    }
    //}
    // Create device buffers
    auto num_det = engine->getMaxDetections();
    //cout << "###datasize:" << data.size() << "input1: " << inputSize[0] <<
    //     " intput2" << inputSize[1] << " det:" << num_det << endl;
    void *&data_d = dataContianer["data"], *&scores_d = dataContianer["score"],
            *&boxes_d = dataContianer["box"], *&classes_d = dataContianer["classes"];
    // Copy image to device
    size_t dataSize = data.size() * sizeof(float);
    cudaMemcpy(data_d, data.data(), dataSize, cudaMemcpyHostToDevice);

    // Run inference n times
    vector<void *> buffers = {data_d, scores_d, boxes_d, classes_d};
    //cout << "Running inference..." << endl;
    engine->infer(buffers, batch);
    //const int count = 100;
    //auto start = chrono::steady_clock::now();
    //for (int i = 0; i < count; i++) {
    //    engine->infer(buffers, batch);
    //}
    //auto stop = chrono::steady_clock::now();
    //auto timing = chrono::duration_cast < chrono::duration < double >> (stop - start);
    //cout << "Took " << timing.count() << " seconds per inference." << endl;
    //cout << "inference over" << endl;
    // Get back the bounding boxes
    cudaMemcpy(scores, scores_d, sizeof(float) * num_det * batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(boxes, boxes_d, sizeof(float) * num_det * 4 * batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(classes, classes_d, sizeof(float) * num_det * batch, cudaMemcpyDeviceToHost);
}

}
//void main(const char *engine_file, char *name, int info[],
//                  float *img_input, float *scores, float *boxes,
//                  float *classes, int batch) {
//    initialize(engine_file, name, info);
//    process(img_input, scores, boxes, classes,
//            batch, name)
//}
