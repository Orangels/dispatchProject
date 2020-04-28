//nvcc -o gn test_onnx.cpp ../cuda/groupnorm.cu /usr/src/tensorrt/samples/common/logger.cpp
// -I/home/user/package/cub-1.8.0 -I/usr/src/tensorrt/samples/common/ -I./../cuda/ -L/usr/local/cuda/lib64
// -lcudart -lcuda -L/usr/local/lib/ -lnvonnxparser -L/usr/lib/x86_64-linux-gnu/
// -lnvinfer  -lnvparsers -lnvinfer_plugin
#include "GN.h"
#include "NvInfer.h"
#include "logger.h"
#include "common.h"
#include "cuda_runtime_api.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>


#define GN_PLUGIN_NAME "group_norm"
#define GN_PLUGIN_VERSION "1"

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 2;
static const int INPUT_W = 2;
static const int INPUT_C = 8;
static const int OUTPUT_SIZE = INPUT_H * INPUT_W * INPUT_C;
//samplesCommon::Args gArgs;

const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;

const std::string gSampleName = "TensorRT.TestLayer";


// Creat the engine using only the API and not any parser.
ICudaEngine *createCustomEngine(unsigned int maxBatchSize, IBuilder *builder, DataType dt) {
    INetworkDefinition *network = builder->createNetwork();

    // Create input tensor of shape {4, 2, 2 } with name INPUT_BLOB_NAME
    ITensor *iptdata = network->addInput(INPUT_BLOB_NAME, dt, Dims3{INPUT_C, INPUT_H, INPUT_W});
    assert(iptdata);

    // Add decode plugins
    std::cout << "Building accelerated plugins..." << std::endl;
    ITensor *xx[] = {iptdata};
    static const int nbWeights = 2, GROUP = 2;
    float EPS = 1e-5;
    float hostData[INPUT_C] = {0}, biasData[INPUT_C] = {0};
    for (int i = 0; i < INPUT_C; i++)hostData[i] = 1;
////////--1. creater layer by IPluginCreator--/////////////
//    std::vector <nvinfer1::PluginField> Attr;
//    Attr.emplace_back(nvinfer1::PluginField("count", &INPUT_C, nvinfer1::PluginFieldType::kINT32, 1));
//    Attr.emplace_back(nvinfer1::PluginField("num_groups", &GROUP, nvinfer1::PluginFieldType::kINT32, 1));
//    Attr.emplace_back(nvinfer1::PluginField("eps", &EPS, nvinfer1::PluginFieldType::kFLOAT32, 1));
//    Attr.emplace_back(nvinfer1::PluginField("w", hostData, nvinfer1::PluginFieldType::kFLOAT32, INPUT_C));
//    Attr.emplace_back(nvinfer1::PluginField("b", biasData, nvinfer1::PluginFieldType::kFLOAT32, INPUT_C));
//    nvinfer1::PluginFieldCollection mFC = {int(Attr.size()), Attr.data()};
//
//    auto creator = getPluginRegistry()->getPluginCreator(GN_PLUGIN_NAME, GN_PLUGIN_VERSION);
//    nvinfer1::IPluginV2 *gnPlugin = creator->createPlugin("", &mFC);
//    auto gnlayer = network->addPluginV2(xx, 1, *gnPlugin);
//    assert(gnlayer);
/////////////////////

////////--2. creater layer by IPlugin--/////////////
    Weights mKernelWeights = Weights{DataType::kFLOAT, hostData, INPUT_C},
            mBiasWeights = Weights{DataType::kFLOAT, biasData, INPUT_C};

    const nvinfer1::Weights weights[] = {mKernelWeights, mBiasWeights};
    auto gnPlugin = GNPlugin(weights, nbWeights, GROUP, EPS);
    auto gnlayer = network->addPluginV2(xx, 1, gnPlugin);
    assert(gnlayer);
/////////////////////

//###################
//    IActivationLayer *layer = network->addActivation(*gnlayer->getOutput(0), ActivationType::kRELU);
//    assert(layer);
//###################

    auto nbOutputs = gnlayer->getNbOutputs();
    auto output = gnlayer->getOutput(0);
    std::cout << "nbOutputs: " << nbOutputs << "INPUT_C: " << INPUT_C;
    gnlayer->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*gnlayer->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
    ICudaEngine *engine = builder->buildCudaEngine(*network);
    network->destroy();
    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream) {
    // Create builder
    IBuilder *builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = createCustomEngine(maxBatchSize, builder, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext &context, float *input, float *output, int batchSize) {
    const ICudaEngine &engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost,
                          stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

//!
//! \brief This function prints the help information for running this sample
//!
void printHelpInfo() {
    std::cout
            << "Usage: ./sample_mnist_api [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]\n";
    std::cout << "--help          Display help information\n";
    std::cout
            << "--datadir       Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)"
            << std::endl;
    std::cout
            << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform."
            << std::endl;
    std::cout << "--int8          Run in Int8 mode.\n";
    std::cout << "--fp16          Run in FP16 mode.\n";
}


int main(int argc, char **argv) {
    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char **>(argv));
    gLogger.reportTestStart(sampleTest);
    // create a model using the API directly and serialize it to a stream
    IHostMemory *modelStream{nullptr};

    float data[] = {6.5486, 5.9743, 2.1141, 9.3370, 1.3765, 1.5005, 0.8672, 1.6151, 2.5169,
                    4.6985, 4.7457, 3.7380, 1.8509, 0.4440, 2.9834, 7.2476, 6.2173, 8.6723,
                    2.8924, 7.9829, 5.6329, 7.9505, 7.6445, 4.0675, 6.2665, 6.3810, 3.4424,
                    1.3675, 5.4805, 0.8932, 9.6199, 5.3145, 0.4758, 7.7327, 6.0537, 0.4564,
                    1.0053, 4.2043, 0.6361, 0.9264, 6.7714, 1.7562, 6.7792, 5.6586, 5.1015,
                    1.4030, 9.1401, 9.5983, 9.4102, 3.1274, 6.6962, 1.6425, 1.1220, 9.6301,
                    8.8801, 4.8154, 1.0288, 6.0371, 3.6395, 6.8811, 8.1933, 9.6004, 3.5542,
                    5.4013};

    int batch = 2, maxbatch = (2 + 4 - 1) / 4 * 4;
    APIToModel(maxbatch, &modelStream);
    assert(modelStream != nullptr);
    IRuntime *runtime = createInferRuntime(gLogger.getTRTLogger());
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(modelStream->data(), modelStream->size(), nullptr);
    assert(engine != nullptr);
    modelStream->destroy();
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    float prob[OUTPUT_SIZE];
    doInference(*context, data, prob, batch);

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    gLogInfo << "Output:\n";
    for (unsigned int i = 0; i < batch * OUTPUT_SIZE; i++)gLogInfo << prob[i] << ",";
    gLogInfo << std::endl;
}
