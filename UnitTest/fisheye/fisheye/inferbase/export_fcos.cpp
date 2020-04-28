#include <iostream>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <cassert>

#include "../csrc/engine.h"

using namespace std;

// Sample program to build a TensorRT Engine from an ONNX model from RetinaNet
//
// By default TensorRT will target FP16 precision (supported on Pascal, Volta, and Turing GPUs)
//
// You can optionally provide an INT8CalibrationTable file created during RetinaNet INT8 calibration
// to build a TensorRT engine with INT8 precision

//./export /home/user/weight/module.onnx aa.plan
void readTxt(string file, vector <string> &res) {
    ifstream infile;
    infile.open(file.data());   //将文件流对象与文件连接起来
    assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行

    string s;
    while (getline(infile, s)) {
//        cout << s << endl;
        res.push_back(s);
    }
    infile.close();             //关闭文件输入流
}

int main(int argc, char *argv[]) {
    if (argc != 3 && argc != 4) {
        cerr << "Usage: " << argv[0] << " core_model.onnx engine.plan {Int8CalibrationTable}" << endl;
        return 1;
    }

    ifstream onnxFile;
    onnxFile.open(argv[1], ios::in | ios::binary);

    if (!onnxFile.good()) {
        cerr << "\nERROR: Unable to read specified ONNX model " << argv[1] << endl;
        return -1;
    }

    onnxFile.seekg(0, onnxFile.end);
    size_t size = onnxFile.tellg();
    onnxFile.seekg(0, onnxFile.beg);

    auto *buffer = new char[size];
    onnxFile.read(buffer, size);
    onnxFile.close();

    // Define default RetinaNet parameters to use for TRT export
    int batch = 8;
    float score_thresh = 0.05f;
    int top_n = 1000;
    size_t workspace_size = (1ULL << 30);
    float nms_thresh = 0.5;
    int detections_per_im = 100;
    bool verbose = false;
    vector<int> strides = {16, 32, 64, 128, 256};

    // For now, assume we have already done calibration elsewhere
    // if we want to create an INT8 TensorRT engine, so no need
    // to provide calibration files or model name
    vector <string> calibration_files;
    readTxt("/home/user/project/fisheye/weights/calibration_files.txt", calibration_files);
    if (argc == 4)
        for (int i = 0; i < 5; i++)cout << calibration_files[i] << endl;
    else
        calibration_files.clear();//https://blog.csdn.net/a272846945/article/details/51182144
    string model_name = "Fcos";
    string calibration_table = argc == 4 ? string(argv[3]) : "";

    // Use FP16 precision by default, use INT8 if calibration table is provided
    string precision = "FP16";
    if (argc == 4)
        precision = "INT8";
    cout << "generate plan info..." << endl;
    cout << "   batch: " << batch << endl;
    cout << "   precision: " << precision << endl;
    cout << "   nms_thresh: " << nms_thresh << endl;
    cout << "   score_thresh: " << score_thresh << endl;
    cout << "   detections_per_im: " << detections_per_im << endl;
    cout << "   calibration_files size: " << calibration_files.size() << endl;
    cout << "Building engine..." << endl;
    auto engine = def_retinanet::Engine(buffer, size, batch, precision, score_thresh, top_n, strides,
                                            nms_thresh, detections_per_im, calibration_files, model_name,
                                            calibration_table, verbose, workspace_size);
    engine.save(string(argv[2]));
    auto inputSize = engine.getInputSize();
    std::cout << "H*W:" << inputSize[0] << 'x' << inputSize[1] << std::endl;

    delete[] buffer;
    return 0;
}
