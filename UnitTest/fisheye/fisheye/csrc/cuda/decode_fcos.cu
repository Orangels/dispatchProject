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
//#define  CMAKE_CUDA_FLAGS "--expt-extended-lambda -std=c++11"
#include "decode_fcos.h"
#include "utils.h"

#include <algorithm>
#include <cstdint>

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/tabulate.h>
#include <thrust/count.h>
#include <thrust/find.h>
#include <thrust/system/cuda/detail/cub/device/device_radix_sort.cuh>
#include <thrust/system/cuda/detail/cub/iterator/counting_input_iterator.cuh>

namespace def_retinanet {
namespace cuda {

void cudasafes(cudaError_t code, const char *message, const char *file, int line) {
    if (code != cudaSuccess)
        fprintf(stderr, "CUDA Error:%s. %s. In %s line %d\n",
                cudaGetErrorString(code), message, file, line);
}

template<typename T>
void checkDeviceDataS(int N, const T *B, const char *message) {
    int pl = N * sizeof(T);
    T b[N];
    cudasafes(cudaMemcpy(b, B, pl, cudaMemcpyDeviceToHost), "CheckDeviceData", __FILE__, __LINE__);
    std::cout << message << " :";
    for (int i = 0; i < N; i++)std::cout << b[i] << ',';
    std::cout << std::endl;
}

int decode(int batch_size,
          const void *const *inputs, void **outputs,
          size_t height, size_t width,
          size_t dense_points, size_t num_classes,
          float score_thresh, int top_n, int stride,
          void *workspace, size_t workspace_size, cudaStream_t stream) {

  int scores_size = dense_points * num_classes * height * width;

  if (!workspace || !workspace_size) {
    // Return required scratch space size cub style
//    workspace_size  = get_size_aligned<float>(anchors.size()); // anchors
    workspace_size += get_size_aligned<bool>(scores_size);     // flags
    workspace_size += get_size_aligned<int>(scores_size);      // indices
    workspace_size += get_size_aligned<int>(scores_size);      // indices_sorted
    workspace_size += get_size_aligned<float>(scores_size);    // scores
    workspace_size += get_size_aligned<float>(scores_size);    // scores_sorted

    size_t temp_size_flag = 0;
    thrust::cuda_cub::cub::DeviceSelect::Flagged((void *)nullptr, temp_size_flag,
      thrust::cuda_cub::cub::CountingInputIterator<int>(scores_size),
      (bool *)nullptr, (int *)nullptr, (int *)nullptr, scores_size);
    size_t temp_size_sort = 0;
    thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending((void *)nullptr, temp_size_sort,
      (float *)nullptr, (float *)nullptr, (int *)nullptr, (int *)nullptr, scores_size);
    workspace_size += std::max(temp_size_flag, temp_size_sort);

    return workspace_size;
  }

  auto on_stream = thrust::cuda::par.on(stream);

  auto flags = get_next_ptr<bool>(scores_size, workspace, workspace_size);
  auto indices = get_next_ptr<int>(scores_size, workspace, workspace_size);
  auto indices_sorted = get_next_ptr<int>(scores_size, workspace, workspace_size);
  auto scores = get_next_ptr<float>(scores_size, workspace, workspace_size);
  auto scores_sorted = get_next_ptr<float>(scores_size, workspace, workspace_size);

  for (int batch = 0; batch < batch_size; batch++) {
    //sigmoid
    auto in_scores = static_cast<const float *>(inputs[0]) + batch * scores_size;
    auto in_boxes = static_cast<const float *>(inputs[1]) + batch * (scores_size / num_classes) * 4;
    //sigmoid & * sigmoid（clsScore）
    auto in_centerness = static_cast<const float *>(inputs[2]) + batch * (scores_size / num_classes);

    auto out_scores = static_cast<float *>(outputs[0]) + batch * top_n;
    auto out_boxes = static_cast<float4 *>(outputs[1]) + batch * top_n;
    auto out_classes = static_cast<float *>(outputs[2]) + batch * top_n;

    // Discard scores below threshold
    //http://on-demand.gputechconf.com/gtc/2012/presentations/S0602-GTC2012-Thrust-Parallel-Library.pdf
    //https://thrust.github.io/doc/group__transformations_ga281b2e453bfa53807eda1d71614fb504.html#ga281b2e453bfa53807eda1d71614fb504
    //将大于score_thresh的结果标记，放入flags中，迭代的区间是对第2、3实参的迭代
    thrust::transform(on_stream, in_scores, in_scores + scores_size,
      flags, thrust::placeholders::_1 > score_thresh);

    int *num_selected = reinterpret_cast<int *>(indices_sorted);
    //https://nvlabs.github.io/cub/structcub_1_1_device_select.html#ad1273ce1f20e442c5a045e0fa17fd5fb
    //将flags中的结果，从第三个实参中取值，放入indices中，scores_size是实际的运算的长度，indices是下标
    thrust::cuda_cub::cub::DeviceSelect::Flagged(workspace, workspace_size,
      thrust::cuda_cub::cub::CountingInputIterator<int>(0),
      flags, indices, num_selected, scores_size, stream);
    cudaStreamSynchronize(stream);
    int num_detections = *thrust::device_pointer_cast(num_selected);

    // Only keep top n scores
    auto indices_filtered = indices;
    if (num_detections > top_n) {
      //https://blog.csdn.net/seamanj/article/details/82976687
      //根据indices的结果，将in_scores的值取出放入scores中去
      thrust::gather(on_stream, indices, indices + num_detections,
        in_centerness, scores);
      //http://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html#a44ed1342b9fd17bec3733f10d4e1ae54
      //根据scores排序，将结果写入scores_sorted；所对应的下标indices，也重排后写入indices_sorted，
      //而且这些下标的有效取值在[0，scores_size]之间
      thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(workspace, workspace_size,
        scores, scores_sorted, indices, indices_sorted, num_detections, 0, sizeof(*scores)*8, stream);
        indices_filtered = indices_sorted;
        num_detections = top_n;
    }

    // Gather boxes
    thrust::transform(on_stream, indices_filtered, indices_filtered + num_detections,
      thrust::make_zip_iterator(thrust::make_tuple(out_scores, out_boxes, out_classes)),
      [=] __device__ (int i) {
        int x = i % width;
        int y = (i / width) % height;
        int a = (i / num_classes / height / width) % dense_points;
        int cls = (i / height / width) % num_classes;
        float4 box = float4{
          //对回归box取值
          in_boxes[((a * 4 + 0) * height + y) * width + x],
          in_boxes[((a * 4 + 1) * height + y) * width + x],
          in_boxes[((a * 4 + 2) * height + y) * width + x],
          in_boxes[((a * 4 + 3) * height + y) * width + x]
        };
        float cenTx = x * stride + 0.5f * stride, cenTy = y * stride + 0.5f * stride;

        box = float4{
          max(0.0f,  cenTx - box.x ),
          max(0.0f,  cenTy - box.y ),
          min( cenTx + box.z , width * stride  - 1.0f),
          min( cenTy + box.w , height * stride  - 1.0f)
        };

        return thrust::make_tuple(sqrt(in_centerness[i]), box, cls + 1);
      });

    // Zero-out unused scores
    if (num_detections < top_n) {
      thrust::fill(on_stream, out_scores + num_detections,
        out_scores + top_n, 0.0f);
      thrust::fill(on_stream, out_classes + num_detections,
        out_classes + top_n, 0.0f);
    }
  }

  return 0;
}

}
}
