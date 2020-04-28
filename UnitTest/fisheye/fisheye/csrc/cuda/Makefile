nvcc -o gn groupnorm.cu  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcuda -I/home/user/package/cub-1.8.0

[unable to include cub](https://devtalk.nvidia.com/default/topic/1035921/cuda-programming-and-performance/unable-to-include-cub/)


nvcc -o gn groupnorm.cu -I/usr/local/cuda/targets/x86_64-linux/include/thrust/system/cuda/detail -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcuda


include/thrust/system/cuda/detail/cub/warp/warp_reduce.cuh & targets/x86_64-linux/include/thrust/system/cuda/detail/cub/block/block_scan.cuh