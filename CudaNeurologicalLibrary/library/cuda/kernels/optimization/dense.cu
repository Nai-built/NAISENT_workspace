#include "../../optimization.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void default_dense__applyKernel(const int inputSize, const int outputSize
    , kernel_float* weights, kernel_float* biases
    , const kernel_float* weights_gradient, const kernel_float* biases_gradient

    , const kernel_float multiplier)
{
    int x = blockIdx.x*TILE + threadIdx.x; // output position
    
    if (x < outputSize) {
        biases[x] -= biases_gradient[x] * multiplier;
        for (int y = 0; y < inputSize; ++y) {
            weights[y*outputSize + x] -= weights_gradient[y*outputSize + x] * multiplier;
        }
    }
}

extern "C" void default_dense__apply(const int& inputSize, const int& outputSize
    , kernel_float* weights, kernel_float* biases
    , const kernel_float* weights_gradient, const kernel_float* biases_gradient

    , const kernel_float& multiplier)
{
    dim3 threads(TILE);
    dim3 blocks(
        (outputSize + TILE - 1) / TILE
    );
    // launch kernel
    default_dense__applyKernel<<<blocks, threads>>>(
        inputSize, outputSize,
        weights, biases,
        weights_gradient, biases_gradient,

        multiplier
    );
    cudaDeviceSynchronize();
}