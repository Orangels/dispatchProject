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

#include <NvInfer.h>

#include <cassert>
#include <vector>

#include "../cuda/decode_fcos.h"

using namespace nvinfer1;

#define RETINANET_PLUGIN_NAME "RetinaFcosNetDecode"
#define RETINANET_PLUGIN_VERSION "1"
#define RETINANET_PLUGIN_NAMESPACE ""

namespace def_retinanet {

class DecodeFcosPlugin : public IPluginV2 {
  float _score_thresh;
  int _top_n, _stride;
//  std::vector<int> _stride;
//  float _scale_h,_scale_w;

  size_t _height;
  size_t _width;
  size_t _dense_points;
  size_t _num_classes;

protected:
  void deserialize(void const* data, size_t length) {
    const char* d = static_cast<const char*>(data);
    read(d, _score_thresh);
    read(d, _top_n);
    read(d, _stride);
//    size_t stride_size;
//    read(d, stride_size);
//    while( stride_size-- ) {
//      float val;
//      read(d, val);
//      _stride.push_back(val);
//    }
//    read(d, _scale_h);
//    read(d, _scale_w);
    read(d, _height);
    read(d, _width);
    read(d, _dense_points);
    read(d, _num_classes);
  }

  size_t getSerializationSize() const override {
    return sizeof(_score_thresh) + sizeof(_top_n) + sizeof(_stride)
      + sizeof(_height) + sizeof(_width) + sizeof(_dense_points) + sizeof(_num_classes);
//      + sizeof(_scale_h) + sizeof(_scale_w)
  }

  void serialize(void *buffer) const override {
    char* d = static_cast<char*>(buffer);
    write(d, _score_thresh);
    write(d, _top_n);
    write(d, _stride);
//    write(d, _stride.size());
//    for( auto &val : _stride ) {
//      write(d, val);
//    }
//    write(d, _scale_h);
//    write(d, _scale_w);
    write(d, _height);
    write(d, _width);
    write(d, _dense_points);
    write(d, _num_classes);
  }

public:
  DecodeFcosPlugin(float score_thresh, int top_n, int stride )
    : _score_thresh(score_thresh), _top_n(top_n) ,_stride(stride){}

  DecodeFcosPlugin(void const* data, size_t length) {
      this->deserialize(data, length);
  }

  const char *getPluginType() const override {
    return RETINANET_PLUGIN_NAME;
  }

  const char *getPluginVersion() const override {
    return RETINANET_PLUGIN_VERSION;
  }

  int getNbOutputs() const override {
    return 3;
  }

  Dims getOutputDimensions(int index,
                                     const Dims *inputs, int nbInputDims) override {
    assert(nbInputDims == 3);
    assert(index < this->getNbOutputs());
    return Dims3(_top_n * (index == 1 ? 4 : 1), 1, 1);
  }

  bool supportsFormat(DataType type, PluginFormat format) const override {
    return type == DataType::kFLOAT && format == PluginFormat::kNCHW;
  }

  void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims,
                        int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override {
    assert(type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kNCHW);
    assert(nbInputs == 3);
    auto const& scores_dims = inputDims[0];
    auto const& boxes_dims = inputDims[1];
    auto const& center_dims = inputDims[2];
    assert(scores_dims.d[1] == boxes_dims.d[1]);//h
    assert(scores_dims.d[2] == boxes_dims.d[2]);//w
    assert(center_dims.d[1] == boxes_dims.d[1]);//h
    assert(center_dims.d[2] == boxes_dims.d[2]);//w
    _height = scores_dims.d[1];
    _width = scores_dims.d[2];
    _dense_points = boxes_dims.d[0] / 4;
    _num_classes = scores_dims.d[0] / _dense_points;
  }

  int initialize() override { return 0; }

  void terminate() override {}

  size_t getWorkspaceSize(int maxBatchSize) const override {
    static int size = -1;
    if (size < 0) {
      size = cuda::decode(maxBatchSize, nullptr, nullptr, _height, _width,
              _dense_points, _num_classes, _score_thresh, _top_n, _stride,
        nullptr, 0, nullptr);
    }
    return size;
  }

  int enqueue(int batchSize,
              const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override {
    return cuda::decode(batchSize, inputs, outputs, _height, _width,
      _dense_points, _num_classes, _score_thresh, _top_n, _stride,
      workspace, getWorkspaceSize(batchSize), stream);
  }

  void destroy() override {};

  const char *getPluginNamespace() const override {
    return RETINANET_PLUGIN_NAMESPACE;
  }

  void setPluginNamespace(const char *N) override {

  }

  IPluginV2 *clone() const override {
    return new DecodeFcosPlugin(_score_thresh, _top_n, _stride);//, _scale_h, _scale_w
  }

private:
  template<typename T> void write(char*& buffer, const T& val) const {
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
  }

  template<typename T> void read(const char*& buffer, T& val) {
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
  }
};

class DecodeFcosPluginCreator : public IPluginCreator {
public:
  DecodeFcosPluginCreator() {}

  const char *getPluginName () const override {
    return RETINANET_PLUGIN_NAME;
  }

  const char *getPluginVersion () const override {
    return RETINANET_PLUGIN_VERSION;
  }

  const char *getPluginNamespace() const override {
    return RETINANET_PLUGIN_NAMESPACE;
  }

  IPluginV2 *deserializePlugin (const char *name, const void *serialData, size_t serialLength) override {
    return new DecodeFcosPlugin(serialData, serialLength);
  }

  void setPluginNamespace(const char *N) override {}
  const PluginFieldCollection *getFieldNames() override { return nullptr; }
  IPluginV2 *createPlugin (const char *name, const PluginFieldCollection *fc) override { return nullptr; }
};

REGISTER_TENSORRT_PLUGIN(DecodeFcosPluginCreator);

}

#undef RETINANET_PLUGIN_NAME
#undef RETINANET_PLUGIN_VERSION
#undef RETINANET_PLUGIN_NAMESPACE
