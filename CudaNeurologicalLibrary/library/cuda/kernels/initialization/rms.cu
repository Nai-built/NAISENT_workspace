#include "../../initialization.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <iostream>

__global__ void rms_initializationKernel(const int tensorSize
    , kernel_float* gamma

    , unsigned long initializationSeed)
{
    int x = blockIdx.x*TILE + threadIdx.x; // output position
    
    if (x < tensorSize) {
        gamma[x] = 1.0f;
    }
}

extern "C" void rms__initialization(const int& tensorSize
    , kernel_float* gamma

    , unsigned long initializationSeed)
{
    dim3 threads(TILE);
    dim3 blocks(
        (tensorSize + TILE - 1) / TILE
    );
    rms_initializationKernel<<<blocks, threads>>>(
        tensorSize,
        gamma,

        initializationSeed
    );
    cudaDeviceSynchronize();
}