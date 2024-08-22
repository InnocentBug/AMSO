#include <cuda.h>
#include <stdio.h>
#include "platform_status.cuh"

__global__ void cuda_kernel() {
  const int i= threadIdx.x;
  printf("running CUDA kernel %d\n", i);
}

void cudaPrintPlatformInfo() {
    cuda_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
