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
#include <cassert>
#include "../csrc/engine.h"

using namespace std;
using namespace cv;
extern "C" {
int fun(float *data, int w, int h, char *name, float *out_boxes = nullptr) {
    cout << "Loading engine..." << endl;
    auto engine = def_retinanet::Engine(name);
    auto inputSize = engine.getInputSize();
    assert(inputSize[0] == h && inputSize[1] == w);

    // Create device buffers
    void *data_d, *scores_d, *boxes_d, *classes_d;
    auto num_det = engine.getMaxDetections();
    num_det = num_det < 100 ? num_det : 100;
    cudaMalloc(&data_d, 3 * inputSize[0] * inputSize[1] * sizeof(float));
    cudaMalloc(&scores_d, num_det * sizeof(float));
    cudaMalloc(&boxes_d, num_det * 4 * sizeof(float));
    cudaMalloc(&classes_d, num_det * sizeof(float));

    // Copy image to device
    size_t dataSize = 3 * inputSize[0] * inputSize[1] * sizeof(float);
    cudaMemcpy(data_d, data, dataSize, cudaMemcpyHostToDevice);

    // Run inference n times
    cout << "Running inference..." << endl;
    const int count = 100;
    auto start = chrono::steady_clock::now();
    vector<void *> buffers = {data_d, scores_d, boxes_d, classes_d};
    for (int i = 0; i < count; i++) {
        engine.infer(buffers, 1);
    }
    auto stop = chrono::steady_clock::now();
    auto timing = chrono::duration_cast < chrono::duration < double >> (stop - start);
    cout << "Took " << timing.count() / count * 1000.0 << " ms per inference." << endl;

    // Get back the bounding boxes
    auto scores = new float[num_det];
    auto boxes = new float[num_det * 4];
    auto classes = new float[num_det];
    cudaMemcpy(scores, scores_d, sizeof(float) * num_det, cudaMemcpyDeviceToHost);
    cudaMemcpy(boxes, boxes_d, sizeof(float) * num_det * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(classes, classes_d, sizeof(float) * num_det, cudaMemcpyDeviceToHost);
//    float scale_w = 1.0648, scale_h = 1.065;
    float scale_w = 1, scale_h = 1.0;

    for (int i = 0, j = 0; i < num_det; i++) {
        // Show results over confidence threshold
        if (scores[i] >= 0.00005f) {
            float x1 = boxes[i * 4 + 0] * scale_w;
            float y1 = boxes[i * 4 + 1] * scale_h;
            float x2 = boxes[i * 4 + 2] * scale_w;
            float y2 = boxes[i * 4 + 3] * scale_w;
//            cout << "Found box {" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "| " << x2 - x1 << ", " << y2 - y1
//                 << "} with score " << scores[i] << " and class " << classes[i] << endl;
            if (scores[i] <= 0.25f)continue;
            out_boxes[j++] = x1;
            out_boxes[j++] = y1;
            out_boxes[j++] = x2;
            out_boxes[j++] = y2;
        }
    }

    delete[] scores, boxes, classes;


    return 0;
}
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " image.jpg" << endl;
        return 1;
    }
    cout << "Preparing data..." << endl;
    auto image = imread(argv[1], IMREAD_COLOR);
    std::cout << "img shape:" << image.size() << std::endl;
    int w = image.cols, h = image.rows;
//    auto inputSize = engine.getInputSize();
//    cv::resize(image, image, Size(inputSize[1], inputSize[0]));
//    std::cout << "img shape:" << image.size() << std::endl;
    cv::Mat pixels;
    image.convertTo(pixels, CV_32FC3, 1.0, 0);

    int channels = 3;
    vector<float> img;
    vector<float> data(channels * w * h), outbox(4 * 100);

    if (pixels.isContinuous())
        img.assign((float *) pixels.datastart, (float *) pixels.dataend);
    else {
        cerr << "Error reading image " << argv[1] << endl;
        return -1;
    }

    vector<float> mean{102.9801, 115.9465, 122.7717};
    vector<float> std{1.0, 1.0, 1.0};

    for (int c = 0; c < channels; c++) {
        for (int j = 0, hw = w * h; j < hw; j++) {
            data[c * hw + j] = (img[channels * j + c] - mean[c]) / std[c];
        }
    }
    fun(data.data(), w, h, argv[2], outbox.data());
    for (unsigned int i = 0; i < outbox.size(); i += 4) {
        float x1 = outbox[i + 0], y1 = outbox[i + 1], x2 = outbox[i + 2], y2 = outbox[i + 3];
        cv::rectangle(image, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 255, 0));
    }
    // Write image
    imwrite("detections.png", image);

    return 0;
}