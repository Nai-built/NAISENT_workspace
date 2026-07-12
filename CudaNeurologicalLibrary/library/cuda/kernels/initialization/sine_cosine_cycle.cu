#include "../../initialization.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <iostream>

__global__ void sine_cosine_cycle__initializationKernel(const int embeddingSize
    , kernel_float* __restrict__ sine_multipliers
    , kernel_float* __restrict__ cosine_multipliers)
{
    int x = blockIdx.x*TILE + threadIdx.x; // output position
    
    if (x < embeddingSize) {
        if (x % 2 == 0) {
            // cosine
            const int eCosine = x/2;
            kernel_float denom = pow(10000.0, (2.0 * eCosine) / embeddingSize);
            kernel_float angleMultiplier = 1 / denom;

            cosine_multipliers[eCosine] = angleMultiplier;
        } else {
            // sine
            const int eSine = (x+1)/2;
            kernel_float denom = pow(10000.0, (2.0 * eSine) / embeddingSize);
            kernel_float angleMultiplier = 1 / denom;

            sine_multipliers[eSine] = angleMultiplier;
        }
    }
}

extern "C" void sine_cosine_cycle__initialization(const int& embeddingSize
    , kernel_float* sine_multipliers
    , kernel_float* cosine_multipliers)
{
    dim3 threads(TILE);
    dim3 blocks(
        (embeddingSize + TILE - 1) / TILE
    );
    
    sine_cosine_cycle__initializationKernel<<<blocks, threads>>>(
        embeddingSize,
        sine_multipliers, cosine_multipliers
    );
    cudaDeviceSynchronize();
}