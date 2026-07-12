#include "../../../adjustments.cuh"

#include <iostream>

__global__ void linear_af__gradientKernel(const kernel_float* __restrict__ tensor_gradient
    , kernel_float* __restrict__ biases_gradient
    
    , const int samplesAmount, const int tensorSize)
{
    int x = blockIdx.x*TILE + threadIdx.x; // tensor position
    
    if (x < tensorSize) {
        kernel_float b_g = 0.0f;
        for (int y = 0; y < samplesAmount; ++y) {
            b_g += tensor_gradient[y*tensorSize + x];
        }
        biases_gradient[x] += b_g;
    }
}

extern "C" void linear_af__gradient(const kernel_float* tensor_gradient
    , kernel_float* biases_gradient
    , const int& samplesAmount, const int& tensorSize)
{
    dim3 threads(TILE);
    dim3 blocks(
        (tensorSize + TILE - 1) / TILE
    );
    // launch kernel
    linear_af__gradientKernel<<<blocks, threads>>>(
        tensor_gradient,
        biases_gradient,
        samplesAmount, tensorSize
    );
}