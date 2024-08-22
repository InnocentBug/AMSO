#include "platform_status.cuh"
#include <cuda.h>
#include <stdio.h>

__global__ void cuda_kernel() {
  const int i = threadIdx.x;
  printf("running CUDA kernel %d\n", i);
}

void cudaPrintPlatformInfo() {
  cuda_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}
