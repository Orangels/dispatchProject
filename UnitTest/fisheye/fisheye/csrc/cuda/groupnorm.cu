//
// Created by 李大冲 on 2019-10-29.
//
#ifdef __JETBRAINS_IDE__
#define __shared__
//how it works: https://stackoverflow.com/questions/39980645/enable-code-indexing-of-cuda-in-clion/46101718#46101718
#endif

#include "groupnorm.h"
#include "utils.h"
#include <iostream>
#include <numeric>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

//#include <thrust/system/cuda/detail/cub/block/block_reduce.cuh>
void cudasafe(cudaError_t code, const char *message, const char *file, int line) {
    if (code != cudaSuccess)
        fprintf(stderr, "CUDA Error:%s. %s. In %s line %d\n",
                cudaGetErrorString(code), message, file, line);
}

constexpr int CAFFE_CUDA_NUM_THREADS = 128;
template<typename T>
using BlockReduce = cub::BlockReduce<T, CAFFE_CUDA_NUM_THREADS>;


template<typename T>
__global__ void InvStdCUDAKernel(const int N, const T epsilon, const T *var, T *inv_std) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        inv_std[i] = 1 / (var[i] + epsilon);
}

template<typename T>
__global__ void RowwiseMomentsCUDAKernel(const int cols, const T *X, T *mean, T *var) {
    __shared__ typename BlockReduce<T>::TempStorage m_storage;
    __shared__ typename BlockReduce<T>::TempStorage v_storage;
    const T scale = T(1) / static_cast<T>(cols);
    const int r = blockIdx.x;
    T m_val = 0;
    T v_val = 0;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const int X_index = r * cols + c;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
        m_val += __ldg(X + X_index);
        v_val += __ldg(X + X_index) * __ldg(X + X_index);
#else
        m_val += X[X_index];
        v_val += X[X_index] * X[X_index];
#endif
    }
    m_val = BlockReduce<T>(m_storage).Sum(m_val);
    v_val = BlockReduce<T>(v_storage).Sum(v_val);
    if (threadIdx.x == 0) {
        const T mu = m_val * scale;
        mean[r] = mu;
        var[r] = sqrt(v_val * scale - mu * mu);
    }
}

template<typename T>
__global__ void ColwiseMomentsCUDAKernel(
        const int rows,
        const int cols,
        const T *X,
        T *mean,
        T *var) {
    __shared__ typename BlockReduce<T>::TempStorage m_storage;
    __shared__ typename BlockReduce<T>::TempStorage v_storage;
    const T scale = T(1) / static_cast<T>(rows);
    const int c = blockIdx.x;
    T m_val = 0;
    T v_val = 0;
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
        const int X_index = r * cols + c;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
        m_val += __ldg(X + X_index);
        v_val += __ldg(X + X_index) * __ldg(X + X_index);
#else
        m_val += X[X_index];
        v_val += X[X_index] * X[X_index];
#endif
    }
    m_val = BlockReduce<T>(m_storage).Sum(m_val);
    v_val = BlockReduce<T>(v_storage).Sum(v_val);
    if (threadIdx.x == 0) {
        const T mu = m_val * scale;
        mean[c] = mu;
        var[c] = sqrt(v_val * scale - mu * mu);
    }
}

template<typename T>
__global__ void ComputeFusedParamsCUDAKernel(
        const int N,
        const int G,
        const int K,
        const T *mu,
        const T *rsig,
        const T *gamma,
        const T *beta,
        T *scale,
        T *bias) {

//location: ~/package/pytorch/caffe2/operators/group_norm_op.cu:20
//template<>
//__global__ void ComputeFusedParamsCUDAKernel<float>(
//        const int N,
//        const int G,
//        const int K,
//        const float *mu,
//        const float *rsig,
//        const float *gamma,
//        const float *beta,
//        float *scale,
//        float *bias) {
    const int C = G * K;
    const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
    if (index < N * C) {
        const int ng = index / K;
        const int c = index % C;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
        const float scale_val = __ldg(gamma + c) * __ldg(rsig + ng);
    scale[index] = scale_val;
    bias[index] = fmaf(-scale_val, __ldg(mu + ng), __ldg(beta + c));
#else
        const float scale_val = gamma[c] * rsig[ng];
        scale[index] = scale_val;
        bias[index] = fmaf(-scale_val, mu[ng], beta[c]);
#endif
    }
}


template<typename T>
__global__ void GroupNormForwardCUDAKernel(
        const int N,
        const int C,
        const int HxW,
        const T *X,
        const T *scale,
        const T *bias,
        T *Y);

//location: ~/package/pytorch/caffe2/operators/group_norm_op.cu:60
template<>
__global__ void GroupNormForwardCUDAKernel<float>(
        const int N,
        const int C,
        const int HxW,
        const float *X,
        const float *scale,
        const float *bias,
        float *Y) {
    const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
    if (index < N * C * HxW) {
        const int nc = index / HxW;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
        Y[index] = fmaf(__ldg(X + index), __ldg(scale + nc), __ldg(bias + nc));
#else
        Y[index] = fmaf(X[index], scale[nc], bias[nc]);
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////                  cuda & cpp                  //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
void checkDeviceData(int N, const float *B, const char *message) {
    int pl = N * sizeof(float);
    float b[N];
    cudasafe(cudaMemcpy(b, B, pl, cudaMemcpyDeviceToHost), "CheckDeviceData", __FILE__, __LINE__);
    std::cout << message << " :";
    for (int i = 0; i < N; i++)std::cout << b[i] << ',';
    std::cout << std::endl;
}

bool CheckReduceDims(const int ndim, const int *X_dims, const int *Y_dims) {
    for (int i = 0; i < ndim; ++i) {
        if (X_dims[i] != Y_dims[i] && Y_dims[i] != 1) {
            return false;
        }
    }
    return true;
}

bool IsRowwiseReduce(
        const int ndim,
        const int *A_dims,
        const int *B_dims,
        int *rows,
        int *cols) {
    *cols = 1;
    int pivot = ndim - 1;
    for (; pivot >= 0 && B_dims[pivot] == 1; --pivot) {
        *cols *= A_dims[pivot];
    }
    *rows = 1;
    for (int i = pivot; i >= 0; --i) {
        if (A_dims[i] != B_dims[i]) {
            return false;
        }
        *rows *= A_dims[i];
    }
    return true;
}

bool IsColwiseReduce(
        const int ndim,
        const int *A_dims,
        const int *B_dims,
        int *rows,
        int *cols) {
    *rows = 1;
    int pivot = 0;
    for (; pivot < ndim && B_dims[pivot] == 1; ++pivot) {
        *rows *= A_dims[pivot];
    }
    *cols = 1;
    for (int i = pivot; i < ndim; ++i) {
        if (A_dims[i] != B_dims[i]) {
            return false;
        }
        *cols *= A_dims[i];
    }
    return true;
}

//location:~/package/pytorch/caffe2/utils/math_gpu.cu:2915
template<typename T>
void InvStdCUDA(
        const int N,
        const T epsilon,
        const T *var,
        T *inv_std,
        cudaStream_t stream) {// N * G, static_cast<T>(epsilon_), rsig, rsig, &context_
    const int M = (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    InvStdCUDAKernel<T>
            << < M, CAFFE_CUDA_NUM_THREADS, 0, stream >> >
                                               (N, epsilon, var, inv_std);
}

//location:~/package/pytorch/caffe2/utils/math/reduce.cu:578
template<typename T>
void MomentsCUDA(
        const int ndim,
        const int *X_dims,
        const int *Y_dims,
        const T *X,
        T *mean,
        T *var,
        cudaStream_t stream) {
    if (not CheckReduceDims(ndim, X_dims, Y_dims)) {
        std::cerr << "Invalid Dims!" << std::endl;
        throw "error";
    };
    const int X_size =
            std::accumulate(X_dims, X_dims + ndim, 1, std::multiplies<int>());
    const int Y_size =
            std::accumulate(Y_dims, Y_dims + ndim, 1, std::multiplies<int>());
    if (X_size == 0) {
        cudaMemsetAsync(mean, T(0), sizeof(T) * Y_size, stream);
        cudaMemsetAsync(var, T(0), sizeof(T) * Y_size, stream);
        return;
    }
    if (std::equal(X_dims, X_dims + ndim, Y_dims)) {
//        std::cout << "259 equal";
        cudaMemcpyAsync(
                mean,
                X,
                sizeof(T) * X_size,
                cudaMemcpyDeviceToDevice,
                stream);
        cudaMemsetAsync(var, T(0), sizeof(T) * Y_size, stream);
        return;
    }
    int rows;
    int cols;
    if (IsRowwiseReduce(ndim, X_dims, Y_dims, &rows, &cols)) {
//        std::cout << "271 IsRowwiseReduce,r:" << rows << "Col:" << cols;
        RowwiseMomentsCUDAKernel<T>
                << < rows, CAFFE_CUDA_NUM_THREADS, 0, stream >> > (
                cols, X, mean, var);
        return;
    }
    if (IsColwiseReduce(ndim, X_dims, Y_dims, &rows, &cols)) {
//        std::cout << "278 IsColwiseReduce";
        ColwiseMomentsCUDAKernel<T>
                << < cols, CAFFE_CUDA_NUM_THREADS, 0, stream >> > (
                rows, cols, X, mean, var);
        return;
    }
    std::cerr << "Invalid parameters!" << std::endl;
    throw "error";
}

//template<>GroupNorm<float>::
void ComputeFusedParams(
        const int N,
        const int G,
        const int K,
        const float *mu,
        const float *rsig,
        const float *gamma,
        const float *beta,
        float *scale,
        float *bias,
        cudaStream_t stream) {
    const int M = (N * G * K + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    ComputeFusedParamsCUDAKernel<float>
            << < M, CAFFE_CUDA_NUM_THREADS, 0, stream >> > (
            N, G, K, mu, rsig, gamma, beta, scale, bias);
}


//template<>GroupNorm<float>::
void GroupNormForwardNCHW(
        const int N,
        const int C,
        const int HxW,
        const float *X,
        const float *scale,
        const float *bias,
        float *Y,
        cudaStream_t stream) {
    const int M = (N * C * HxW + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    GroupNormForwardCUDAKernel<float>
            << < M, CAFFE_CUDA_NUM_THREADS, 0, stream >> > (
            N, C, HxW, X, scale, bias, Y);
}

int calWorkSpace(const int N, const int C, const int G) {
    int workspace_size = get_size_aligned<float>(N * C);
    workspace_size += get_size_aligned<float>(N * C);
    workspace_size += get_size_aligned<float>(N * G);
    workspace_size += get_size_aligned<float>(N * G);
    return workspace_size;
}

//template<>GroupNorm<float>::
int RunOnDeviceWithOrderNCHW(
        const int N,
        const int C,
        const int HxW,//X.numel() / (N * C)
        const int G,
        const float *X,//input, with data in them
        const float *gamma,
        const float *beta,
        float *Y,
        float epsilon_,
        void *workspace, size_t workspace_size, cudaStream_t stream) {//it has storage space without data
    const int K = C / G;
//    std::cout << '|' << stream << '|' << __FILE__ << std::endl;
    //    std::cout << "-n" << N << "-c" << C << "-hw" << HxW
//              << "-g" << G << "-e" << epsilon_ << "-k" << K << std::endl;
    const std::array<int, 2> X_dims = {N * G, K * HxW};
    const std::array<int, 2> Y_dims = {N * G, 1};
    int stastistic_size = N * G, affine_size = N * C;
    // already alloc
    auto mu = get_next_ptr<float>(stastistic_size, workspace, workspace_size);
    auto rsig = get_next_ptr<float>(stastistic_size, workspace, workspace_size);
    auto scale_data = get_next_ptr<float>(affine_size, workspace, workspace_size);
    auto bias_data = get_next_ptr<float>(affine_size, workspace, workspace_size);
//    float *mu, *rsig, *scale_data, *bias_data;
//    cudasafe(cudaMalloc((void **) &mu, sizeof(float) * N * G), "-+-mu-+-", __FILE__, __LINE__);
//    cudasafe(cudaMalloc((void **) &rsig, sizeof(float) * N * G), "-+-rsig-+-", __FILE__, __LINE__);
//    cudasafe(cudaMalloc((void **) &scale_data, sizeof(float) * N * C), "-+-scale-+-", __FILE__, __LINE__);
//    cudasafe(cudaMalloc((void **) &bias_data, sizeof(float) * N * C), "-+-bias-+-", __FILE__, __LINE__);
//    float *yy;
//    cudasafe(cudaMalloc((void **) &yy, sizeof(float) * N * C * HxW), "-+-bias-+-", __FILE__, __LINE__);

    MomentsCUDA<float>(2, X_dims.data(), Y_dims.data(), X, mu, rsig, stream);
//    std::cout << "HxW: " << HxW << ",input X: " << std::endl;
//    checkDeviceData(10, X, "X");
//    checkDeviceData(N * G, mu, "mu");
//    checkDeviceData(N * G, rsig, "std");
    InvStdCUDA<float>(N * G, static_cast<float>(epsilon_), rsig, rsig, stream);
//    checkDeviceData(N * G, rsig, "~std");
    ComputeFusedParams(N, G, K, mu, rsig, gamma, beta, scale_data, bias_data, stream);

    GroupNormForwardNCHW(N, C, HxW, X, scale_data, bias_data, Y, stream);
//    int lenY = N * C * HxW * sizeof(float);
//    float *host_a, aaa[5] = {0};
//    cudasafe(cudaHostAlloc((void **) &host_a, lenY, cudaHostAllocDefault),
//             "host A", __FILE__, __LINE__);
//    cudasafe(cudaMemcpyAsync(host_a, aaa, lenY, cudaMemcpyDeviceToHost, stream), "CheckDeviceData", __FILE__, __LINE__);
//    cudasafe(cudaMemcpyAsync(host_a, yy, lenY, cudaMemcpyDeviceToHost, stream), "CheckDeviceData", __FILE__, __LINE__);
//    cudasafe(cudaMemcpyAsync(Y, host_a, lenY, cudaMemcpyHostToDevice, stream), "HostToDevice", __FILE__, __LINE__);
//    cudasafe(cudaMemcpyAsync(aaa, host_a, lenY, cudaMemcpyHostToDevice, stream), "HostToDevice", __FILE__, __LINE__);

//    std::cout << "/////////////////" << std::endl;
//    printf("hosta: %p, yy: %p, Y: %p\n", host_a, yy, Y);
//    printf("hosta: %p, yy: %p, Y: %p\n", &(*host_a), &(*yy), &(*Y));
//    checkDeviceData(16, yy, "yy");
//    checkDeviceData(10, Y, "Y");
//    cudasafe(cudaFreeHost(host_a), "free host", __FILE__, __LINE__);
//    cudasafe(cudaFree(yy), "free cuda", __FILE__, __LINE__);

    return 0;
}


int void_main() {
    int N = 2, C = 2, HxW = 4, G = 2;
    int cl = sizeof(float) * 16, pl = C * sizeof(float);
    float x[] = {-1.0433, -1.1988, -0.3371, 1.3956, 1.4064, 0.7155, -0.5022, 1.7002,
                 -0.2503, 0.1939, -1.7116, -0.4951, 1.5856, 0.7153, 0.5854, -1.7816},
            b[C], w[C];
    for (int i = 0; i < C; i++) w[i] = 1;
    float *X, *Y, *W, *B, *workspace;
    int len = calWorkSpace(N, C, G);
    cudasafe(cudaMalloc((void **) &workspace, len), "workspace", __FILE__, __LINE__);
    cudasafe(cudaMalloc((void **) &X, cl), "x", __FILE__, __LINE__);
    cudasafe(cudaMalloc((void **) &Y, cl), "y", __FILE__, __LINE__);
    cudasafe(cudaMalloc((void **) &W, pl), "w", __FILE__, __LINE__);
    cudasafe(cudaMalloc((void **) &B, pl), "b", __FILE__, __LINE__);
    cudasafe(cudaMemset(B, 0, pl), "~b", __FILE__, __LINE__);
    cudasafe(cudaMemcpy(W, w, pl, cudaMemcpyHostToDevice), "~w", __FILE__, __LINE__);

//    std::cout << "\nworkspace:" << len << std::endl;
//    for (int i = 0; i < C; i++) w[i] = 100;
//    std::cout << "\ny:";
//    for (int i = 0; i < 16; i++)std::cout << y[i] << ',';
//    std::cout << "\nX:";
//    for (int i = 0; i < 16; i++)std::cout << x[i] << ',';

    cudasafe(cudaMemcpy(X, x, cl, cudaMemcpyHostToDevice), "Copy x ", __FILE__, __LINE__);
    cudasafe(cudaMemcpy(w, W, pl, cudaMemcpyDeviceToHost), "Copy w", __FILE__, __LINE__);
    cudasafe(cudaMemcpy(b, B, pl, cudaMemcpyDeviceToHost), "Copy b ", __FILE__, __LINE__);
//    checkDeviceData(16, X, "X");
//    std::cout << "\nW:";
//    for (int i = 0; i < C; i++)std::cout << w[i] << ',';
//    std::cout << "\nB:";
//    for (int i = 0; i < C; i++)std::cout << b[i] << ',';
//    std::cout << "\n;";


//    auto gn = GroupNorm<float>(N, C, G);
    cudaStream_t stream;
    cudaError_t cudaErr = cudaStreamCreate(&stream);
    int ret = RunOnDeviceWithOrderNCHW(N, C, HxW, G, X, W, B, Y, 1e-5, workspace, len, stream);
//    cudasafe(cudaMemcpy(y, Y, cl, cudaMemcpyDeviceToHost), "Copy y ", __FILE__, __LINE__);
//    std::cout << "\nY:";
//    for (int i = 0; i < 16; i++)std::cout << y[i] << ',';
//    std::cout << "\n;";
    checkDeviceData(N * C * HxW, Y, "Y");
    return 0;
}

