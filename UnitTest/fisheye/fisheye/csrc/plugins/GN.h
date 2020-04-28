//
// Created by 李大冲 on 2019-10-29.
//

#ifndef INFER__GN_H
#define INFER__GN_H

#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <cassert>
#include <NvInfer.h>
#include "../cuda/groupnorm.h"

#define GN_PLUGIN_NAME "group_norm"
#define GN_PLUGIN_VERSION "1"
#define GN_PLUGIN_NAMESPACE ""
#define CHECK_CU(status) { if (status != 0) throw std::runtime_error(__FILE__ +  __LINE__ + std::string{"CUDA Error: "} + std::to_string(status)); }


// Helpers to move data to/from the GPU.
nvinfer1::Weights copyToDevice(const void *hostData, int count) {
    void *deviceData;
    CHECK_CU(cudaMalloc(&deviceData, count * sizeof(float)));
    CHECK_CU(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
    return nvinfer1::Weights{nvinfer1::DataType::kFLOAT, deviceData, count};
}

int copyFromDevice(char *hostBuffer, nvinfer1::Weights deviceWeights) {
    *reinterpret_cast<int *>(hostBuffer) = deviceWeights.count;
    CHECK_CU(cudaMemcpy(hostBuffer + sizeof(int), deviceWeights.values, deviceWeights.count * sizeof(float),
                        cudaMemcpyDeviceToHost));
    return sizeof(int) + deviceWeights.count * sizeof(float);
}

template<typename T>
void write(char *&buffer, const T &val) {
    *reinterpret_cast<T *>(buffer) = val;
    buffer += sizeof(T);
}

template<typename T>
void read(const char *&buffer, T &val) {
    val = *reinterpret_cast<const T *>(buffer);
    buffer += sizeof(T);
}

void checkTensorData(int N, const void *inputs, const char *message) {
    const float *B = reinterpret_cast<const float *>(inputs);
    int pl = N * sizeof(float);
    float b[N];
    cudaMemcpy(b, B, pl, cudaMemcpyDeviceToHost);
    std::cout << message << " in " << __FILE__ << "@" << __LINE__ << " :";
    for (int i = 0; i < N; i++)std::cout << b[i] << ',';
    std::cout << std::endl;
}

class GNPlugin : public nvinfer1::IPluginV2 {

public:
    // In this simple case we're going to infer the number of output channels from the bias weights.
    // The knowledge that the kernel weights are weights[0] and the bias weights are weights[1] was
    // divined from the caffe innards
    GNPlugin(const nvinfer1::Weights *weights, int nbWeights, int group, float epsilon) {
        assert(nbWeights == 2);
        mKernelWeights = copyToDevice(weights[0].values, weights[0].count);
        mBiasWeights = copyToDevice(weights[1].values, weights[1].count);
        G = group;
        epsilon_ = epsilon;
    }

    GNPlugin() = delete;

    // Create the plugin at runtime from a byte stream.
    GNPlugin(const void *data, size_t length) {
        const char *d = reinterpret_cast<const char *>(data);
        const char *CHECK_CU = d;
        // Deserialize kernel.
        read(d, C);
        read(d, HxW);
        read(d, G);
        read(d, epsilon_);
        const int kernelCount = reinterpret_cast<const int *>(d)[0];
        mKernelWeights = copyToDevice(d + sizeof(int), kernelCount);
        d += sizeof(int) + mKernelWeights.count * sizeof(float);
        // Deserialize bias.
        const int biasCount = reinterpret_cast<const int *>(d)[0];
        mBiasWeights = copyToDevice(d + sizeof(int), biasCount);
        d += sizeof(int) + mBiasWeights.count * sizeof(float);
        // CHECK_CU that the sizes are what we expected.
        assert(d == CHECK_CU + length);
    }

    virtual int getNbOutputs() const override { return 1; }

    virtual nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims *inputs,
                                               int nbInputDims) override {
        //attention: although input should be NCHW, it's CHW actually
        assert(index == 0 && nbInputDims == 1);
        return *inputs;
    }

    virtual int initialize() override { return 0; }

    virtual void terminate() override {}

    virtual size_t getWorkspaceSize(int maxBatchSize) const override {
        static int space = calWorkSpace(maxBatchSize, C, G);
        return space;
    }

    virtual int enqueue(int batchSize, const void *const *inputs, void **outputs,
                        void *workspace, cudaStream_t stream) override {
        return RunOnDeviceWithOrderNCHW(batchSize, C, HxW, G,
                                        reinterpret_cast<const float *>(inputs[0]),
                                        reinterpret_cast<const float *>(mKernelWeights.values),
                                        reinterpret_cast<const float *>(mBiasWeights.values),
                                        reinterpret_cast<float *>(outputs[0]),
                                        epsilon_,
                                        workspace,
                                        getWorkspaceSize(batchSize),
                                        stream);
    }

    // For this sample, we'll only support float32 with NCHW.
    virtual bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override {
        return (type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kNCHW);
    }

    void configureWithFormat(const nvinfer1::Dims *inputDims, int nbInputs,
                             const nvinfer1::Dims *outputDims, int nbOutputs,
                             nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize) {
        assert(type == nvinfer1::DataType::kFLOAT);
        assert(format == nvinfer1::PluginFormat::kNCHW);
        assert(mKernelWeights.count == inputDims[0].d[0] && mBiasWeights.count == inputDims[0].d[0]);
        assert(nbOutputs == 1 && outputDims[0].d[0] == outputDims[0].d[0]);
        C = inputDims[0].d[0];
        HxW = inputDims[0].d[1] * inputDims[0].d[2];
    }

    size_t getSerializationSize() const override {
        return sizeof(int) * (2 + 3) + sizeof(float) + 2 * C * sizeof(float);
    }

    void serialize(void *buffer) const override {
        char *d = static_cast<char *>(buffer);
        const char *CHECK_CU = d;
        write(d, C);
        write(d, HxW);
        write(d, G);
        write(d, epsilon_);
        d += copyFromDevice(d, mKernelWeights);
        d += copyFromDevice(d, mBiasWeights);
        assert(d == CHECK_CU + getSerializationSize());
    }

    // Free buffers.
    void destroy() override {
        cudaFree(const_cast<void *>(mKernelWeights.values));
        mKernelWeights.values = nullptr;
        cudaFree(const_cast<void *>(mBiasWeights.values));
        mBiasWeights.values = nullptr;
    }

    const char *getPluginType() const override {
        return GN_PLUGIN_NAME;
    }

    const char *getPluginVersion() const override {
        return GN_PLUGIN_VERSION;
    }

    const char *getPluginNamespace() const override {
        return GN_PLUGIN_NAMESPACE;
    }

    void setPluginNamespace(const char *N) override {}

    IPluginV2 *clone() const override {
        const int nbWeights = 2;
        const nvinfer1::Weights weights[nbWeights] = {mKernelWeights, mBiasWeights};
        return new GNPlugin(weights, nbWeights, G, epsilon_);
    }

private:
    int C, HxW, G;
    float epsilon_;
    nvinfer1::Weights mKernelWeights{nvinfer1::DataType::kFLOAT, nullptr},
            mBiasWeights{nvinfer1::DataType::kFLOAT, nullptr};

};

//class GNPluginFactory : public nvinfer1::IPluginFactory {
//public:
//    bool isPlugin(const char *name) override {
//        printf("gn factory: %s", name);
//        return isPluginExt(name);
//    }
//
//    bool isPluginExt(const char *name) override {
//        printf("gn factory: %s", name);
//        return !strcmp(name, GN_PLUGIN_NAME);
//    }
//
//    // Create a plugin using provided weights.
//    virtual nvinfer1::IPlugin *
//    createPlugin(const char *layerName, const nvinfer1::Weights *weights, int nbWeights) override {
//        static const int GROUP = 32, EPS = 1e-5;
//        assert(isPluginExt(layerName) && nbWeights == 2);
//        assert(mPlugin.get() == nullptr);
//        mPlugin = std::unique_ptr<GNPlugin>(new GNPlugin(weights, nbWeights, GROUP, EPS));
//        return mPlugin.get();
//    }
//
//    // Create a plugin from serialized data.
//    virtual nvinfer1::IPlugin *
//    createPlugin(const char *layerName, const void *serialData, size_t serialLength) override {
//        assert(isPlugin(layerName));
//        // This will be automatically destroyed when the engine is destroyed.
//        return new GNPlugin{serialData, serialLength};
//    }
//
//    // User application destroys plugin when it is safe to do so.
//    // Should be done after consumers of plugin (like ICudaEngine) are destroyed.
//    void destroyPlugin() { mPlugin.reset(); }
//
//    std::unique_ptr <GNPlugin> mPlugin{nullptr};
//};

class GNPluginCreator : public nvinfer1::IPluginCreator {
public:
    GNPluginCreator() {
        std::vector <nvinfer1::PluginField> Attributes;
        // Describe ClipPlugin's required PluginField arguments
        Attributes.emplace_back(nvinfer1::PluginField("count", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        Attributes.emplace_back(nvinfer1::PluginField("num_groups", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        Attributes.emplace_back(nvinfer1::PluginField("eps", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
        Attributes.emplace_back(nvinfer1::PluginField("w", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
        Attributes.emplace_back(nvinfer1::PluginField("b", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));

        // Fill PluginFieldCollection with PluginField arguments metadata
        mFC.nbFields = Attributes.size();
        mFC.fields = Attributes.data();
    }

    const char *getPluginName() const override { return GN_PLUGIN_NAME; }

    const char *getPluginVersion() const override { return GN_PLUGIN_VERSION; }

    void setPluginNamespace(const char *N) override { mNamespace = N; }

    const char *getPluginNamespace() const { return mNamespace.c_str(); }

    nvinfer1::IPluginV2 *deserializePlugin(const char *name, const void *serialData,
                                           size_t serialLength) override {
        printf("name is %s\n", name);
        return new GNPlugin(serialData, serialLength);
    }

    const nvinfer1::PluginFieldCollection *getFieldNames() override { return &mFC; }

    nvinfer1::IPluginV2 *createPlugin(const char *name,
                                      const nvinfer1::PluginFieldCollection *fc) override {
        int count, group;
        float eps;
        const float *kernel, *bias;
        const nvinfer1::PluginField *fields = fc->fields;

        // Parse fields from PluginFieldCollection
        assert(fc->nbFields == 5);
        for (int i = 0; i < fc->nbFields; i++) {
            if (strcmp(fields[i].name, "count") == 0) {
                assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                count = *(reinterpret_cast<const int *>(fields[i].data));
            } else if (strcmp(fields[i].name, "num_groups") == 0) {
                assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                group = *(static_cast<const int *>(fields[i].data));
            } else if (strcmp(fields[i].name, "eps") == 0) {
                assert(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
                eps = *(static_cast<const float *>(fields[i].data));
            } else if (strcmp(fields[i].name, "w") == 0) {
                assert(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
                kernel = static_cast<const float *>(fields[i].data);
            } else if (strcmp(fields[i].name, "b") == 0) {
                assert(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
                bias = static_cast<const float *>(fields[i].data);
            }
        }

        nvinfer1::Weights weights[] = {nvinfer1::Weights{nvinfer1::DataType::kFLOAT, kernel, count},
                                       nvinfer1::Weights{nvinfer1::DataType::kFLOAT, bias, count}};
        return new GNPlugin(weights, 2, group, eps);
    }

private:
    nvinfer1::PluginFieldCollection mFC;
    std::string mNamespace;
};

REGISTER_TENSORRT_PLUGIN(GNPluginCreator);

#undef  GN_PLUGIN_NAME
#undef  GN_PLUGIN_VERSION
#undef  GN_PLUGIN_NAMESPACE

#endif //INFER__GN_H
