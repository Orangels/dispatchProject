#include <iostream>
#include <stdexcept>
#include <fstream>
#include <vector>

#include "../csrc/engine.h"

using namespace std;

// Sample program to build a TensorRT Engine from an ONNX model from RetinaNet
//
// By default TensorRT will target FP16 precision (supported on Pascal, Volta, and Turing GPUs)
//
// You can optionally provide an INT8CalibrationTable file created during RetinaNet INT8 calibration
// to build a TensorRT engine with INT8 precision

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
    int batch = 1;
    float score_thresh = 0.05f;
    int top_n = 1000;
    size_t workspace_size = (1ULL << 30);
    float nms_thresh = 0.5;
    int detections_per_im = 100;
    bool verbose = false;
    vector <vector<float>> anchors = {
            {-18.0000,  -8.0000,   25.0000,  15.0000,  -23.7183,  -11.1191,  30.7183,  18.1191,  -30.9228,  -15.0488,  37.9228,  22.0488,  -12.0000,  -12.0000,  19.0000,  19.0000,  -16.1587,  -16.1587,  23.1587,  23.1587,  -21.3984,  -21.3984,  28.3984,  28.3984,  -8.0000,   -20.0000,  15.0000,  27.0000,  -11.1191,  -26.2381,  18.1191,  33.2381,  -15.0488,  -34.0976,  22.0488,  41.0976},
            {-38.0000,  -16.0000,  53.0000,  31.0000,  -49.9564,  -22.2381,  64.9564,  37.2381,  -65.0204,  -30.0976,  80.0204,  45.0976,  -24.0000,  -24.0000,  39.0000,  39.0000,  -32.3175,  -32.3175,  47.3175,  47.3175,  -42.7968,  -42.7968,  57.7968,  57.7968,  -14.0000,  -36.0000,  29.0000,  51.0000,  -19.7183,  -47.4365,  34.7183,  62.4365,  -26.9228,  -61.8456,  41.9228,  76.8456},
            {-74.0000,  -28.0000,  105.0000, 59.0000,  -97.3929,  -39.4365,  128.3929, 70.4365,  -126.8661, -53.8456,  157.8661, 84.8456,  -48.0000,  -48.0000,  79.0000,  79.0000,  -64.6349,  -64.6349,  95.6349,  95.6349,  -85.5937,  -85.5937,  116.5937, 116.5937, -30.0000,  -76.0000,  61.0000,  107.0000, -41.9564,  -99.9127,  72.9564,  130.9127, -57.0204,  -130.0409, 88.0204,  161.0409},
            {-150.0000, -60.0000,  213.0000, 123.0000, -197.3056, -83.9127,  260.3056, 146.9127, -256.9070, -114.0409, 319.9070, 177.0409, -96.0000,  -96.0000,  159.0000, 159.0000, -129.2699, -129.2699, 192.2699, 192.2699, -171.1873, -171.1873, 234.1873, 234.1873, -58.0000,  -148.0000, 121.0000, 211.0000, -81.3929,  -194.7858, 144.3929, 257.7858, -110.8661, -253.7322, 173.8661, 316.7322},
            {-298.0000, -116.0000, 425.0000, 243.0000, -392.0914, -162.7858, 519.0914, 289.7858, -510.6392, -221.7322, 637.6392, 348.7322, -192.0000, -192.0000, 319.0000, 319.0000, -258.5398, -258.5398, 385.5398, 385.5398, -342.3747, -342.3747, 469.3747, 469.3747, -118.0000, -300.0000, 245.0000, 427.0000, -165.3056, -394.6113, 292.3056, 521.6113, -224.9070, -513.8140, 351.9070, 640.8140}
    };

    // For now, assume we have already done calibration elsewhere
    // if we want to create an INT8 TensorRT engine, so no need
    // to provide calibration files or model name
    const vector <string> calibration_files;
    string model_name = "";
    string calibration_table = argc == 4 ? string(argv[3]) : "";

    // Use FP16 precision by default, use INT8 if calibration table is provided
    string precision = "FP16";
    if (argc == 4)
        precision = "INT8";

    cout << "Building engine..." << endl;
    auto engine = def_retinanet::Engine(buffer, size, batch, precision, score_thresh, top_n,
                                        anchors, nms_thresh, detections_per_im, calibration_files, model_name,
                                        calibration_table, verbose, workspace_size);
    engine.save(string(argv[2]));


    delete[] buffer;

    return 0;
}
