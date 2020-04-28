/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <NvInfer.h>

#include <cuda_runtime.h>

using namespace std;
using namespace nvinfer1;

namespace def_retinanet {

// RetinaNet wrapper around TensorRT CUDA engine
class Engine {
public:
    // Create engine from engine path -- void initia
    Engine(const string &engine_path, bool verbose=false);


    // Create engine from serialized onnx model
    Engine(const char *onnx_model, size_t onnx_size, size_t batch, string precision,
           float score_thresh, int top_n, const vector <vector<float>> &anchors,
           float nms_thresh, int detections_per_im, const vector <string> &calibration_files,
           string model_name, string calibration_table, bool verbose, size_t workspace_size = (1ULL << 30));

    // Create engine from serialized onnx model
    Engine(const char *onnx_model, size_t onnx_size, size_t batch, string precision,
           float score_thresh, int top_n, const vector<int> &strides,
           float nms_thresh, int detections_per_im, const vector <string> &calibration_files,
           string model_name, string calibration_table, bool verbose, size_t workspace_size = (1ULL << 30));


    ~Engine();

    // Save model to path
    void save(const string &path);

    // Infer using pre-allocated GPU buffers {data, scores, boxes, classes}
    void infer(vector<void *> &buffers, int batch=1);

    // Get (h, w) size of the fixed input
    vector<int> getInputSize();

    // Get max allowed batch size
    int getMaxBatchSize();

    // Get max number of detections
    int getMaxDetections();

    // Get stride
    int getStride();

    cudaStream_t getStream();

private:
    IRuntime *_runtime = nullptr;
    ICudaEngine *_engine = nullptr;
    IExecutionContext *_context = nullptr;
    cudaStream_t _stream = nullptr;
    void _load(const string &path);
    void _prepare();

};

class Logger : public ILogger {
public:
    Logger(bool verbose)
            : _verbose(verbose) {
    }

    void log(Severity severity, const char *msg) override {
        if (_verbose || (severity != Severity::kINFO))
            std::cout << msg << std::endl;
    }

private:
    bool _verbose{false};
};

}
