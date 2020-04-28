//
// Created by 李大冲 on 2019-10-29.
//

#ifndef INFER__GROUPNORM_H
#define INFER__GROUPNORM_H


int calWorkSpace(const int N, const int C, const int G = 32);

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
        void *workspace, size_t workspace_size, cudaStream_t stream);

//#include <iostream>
//#include <cuda_runtime.h>

/*
//template<typename T>
//class GroupNorm {
//public:
//    GroupNorm(const int N,
//              const int C,
//              const int G = 32) : epsilon_(1e-5), G(G) {
//        cudasafe(cudaMalloc((void **) &mu, sizeof(T) * N * G), "-+-mu-+-", __FILE__, __LINE__);
//        cudasafe(cudaMalloc((void **) &rsig, sizeof(T) * N * G), "-+-rsig-+-", __FILE__, __LINE__);
//        cudasafe(cudaMalloc((void **) &scale_data, sizeof(T) * N * C), "-+-scale-+-", __FILE__, __LINE__);
//        cudasafe(cudaMalloc((void **) &bias_data, sizeof(T) * N * C), "-+-bias-+-", __FILE__, __LINE__);
//        cudaError_t cudaErr = cudaStreamCreate(&stream);
//    }
//
//    ~GroupNorm() {
//        cudaError_t ret = cudaStreamDestroy(stream);
//        cudasafe(cudaFree(mu), "~mu~", __FILE__, __LINE__);
//        cudasafe(cudaFree(rsig), "~rsig~", __FILE__, __LINE__);
//        cudasafe(cudaFree(scale_data), "~scale_data~", __FILE__, __LINE__);
//        cudasafe(cudaFree(bias_data), "~bias_data~", __FILE__, __LINE__);
//    }


    bool RunOnDeviceWithOrderNCHW(
            const int N,
            const int C,
            const int HxW,//X.numel() / (N * C)
            const int G,
            const T *X,//input, with data in them
            const T *gamma,
            const T *beta,
            T *Y);

    void ComputeFusedParams(
            const int N,
            const int G,
            const int K,
            const T *mu,
            const T *rsig,
            const T *gamma,
            const T *beta,
            T *scale,
            T *bias);

    void GroupNormForwardNCHW(
            const int N,
            const int C,
            const int HxW,
            const T *X,
            const T *scale,
            const T *bias,
            T *Y);

private:
    T *mu, *rsig, *scale_data, *bias_data;
    cudaStream_t stream;
    T epsilon_;
    int G;
};
*/

#endif //INFER__GROUPNORM_H
