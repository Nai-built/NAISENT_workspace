#include "../../../adjustments.cuh"

#include <iostream>

__global__ void relu_af__gradientKernel(const kernel_float* __restrict__ tensor_gradient, kernel_float* __restrict__ activation_tensor_gradient
    , kernel_float* __restrict__ biases_gradient
    
    , const int samplesAmount, const int tensorSize

    , const kernel_float* __restrict__ cache_tensor, const kernel_float fadeMultiplier)
{
    int x = blockIdx.x*TILE + threadIdx.x; // tensor position
    
    if (x < tensorSize) {
        kernel_float b_g = 0.0f;
        for (int y = 0; y < samplesAmount; ++y) {
            kernel_float relu_g = tensor_gradient[y*tensorSize + x];
            if (cache_tensor[y*tensorSize + x] < 0) {
                relu_g *= fadeMultiplier;
            }
            b_g += relu_g;
            activation_tensor_gradient[y*tensorSize + x] = relu_g;
        }
        biases_gradient[x] += b_g;
    }
}

extern "C" void relu_af__gradient(const kernel_float* tensor_gradient, kernel_float* activation_tensor_gradient
    , kernel_float* biases_gradient
    , const int& samplesAmount, const int& tensorSize
    , const kernel_float* cache_tensor, const kernel_float& fadeMultiplier)
{
    dim3 threads(TILE);
    dim3 blocks(
        (tensorSize + TILE - 1) / TILE
    );
    // launch kernel
    relu_af__gradientKernel<<<blocks, threads>>>(
        tensor_gradient, activation_tensor_gradient,
        biases_gradient,
        samplesAmount, tensorSize,
        cache_tensor, fadeMultiplier
    );
}