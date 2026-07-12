#include "../../../activations.cuh"

#include <iostream>

__global__ void relu_af__activationKernel(kernel_float* __restrict__ tensor
    , const kernel_float* __restrict__ biases
    
    , const int samplesAmount, const int tensorSize

    , kernel_float* __restrict__ cache_tensor, const kernel_float fadeMultiplier

    , const kernel_float multiplier)
{
    int x = blockIdx.x*TILE + threadIdx.x; // tensor position
    
    if (x < tensorSize) {
        const kernel_float bias = biases[x];
        for (int y = 0; y < samplesAmount; ++y) {
            const kernel_float o = (tensor[y*tensorSize + x] + bias)*multiplier;
            cache_tensor[y*tensorSize + x] = o;
            if (o < 0) {
                tensor[y*tensorSize + x] = o*fadeMultiplier;
            } else {
                tensor[y*tensorSize + x] = o;
            }
        }
    }
}

extern "C" void relu_af__activation(kernel_float* tensor
    , const kernel_float* biases
    , const int& samplesAmount, const int& tensorSize
    , kernel_float* cache_tensor, const kernel_float& fadeMultiplier

    , const kernel_float& multiplier)
{
    dim3 threads(TILE);
    dim3 blocks(
        (tensorSize + TILE - 1) / TILE
    );
    // launch kernel
    relu_af__activationKernel<<<blocks, threads>>>(
        tensor,
        biases,
        samplesAmount, tensorSize,
        cache_tensor, fadeMultiplier,

        multiplier
    );
}