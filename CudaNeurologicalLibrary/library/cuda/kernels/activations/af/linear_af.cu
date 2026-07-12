#include "../../../activations.cuh"

#include <iostream>

__global__ void linear_af__activationKernel(kernel_float* __restrict__ tensor
    , const kernel_float* __restrict__ biases
    
    , const int samplesAmount, const int tensorSize

    , const kernel_float multiplier)
{
    int x = blockIdx.x*TILE + threadIdx.x; // tensor position
    
    if (x < tensorSize) {
        const kernel_float bias = biases[x];
        for (int y = 0; y < samplesAmount; ++y) {
            tensor[y*tensorSize + x] = (tensor[y*tensorSize + x]+bias)*multiplier;
        }
    }
}

extern "C" void linear_af__activation(kernel_float* tensor
    , const kernel_float* biases
    , const int& samplesAmount, const int& tensorSize

    , const kernel_float& multiplier)
{
    dim3 threads(TILE);
    dim3 blocks(
        (tensorSize + TILE - 1) / TILE
    );
    // launch kernel
    linear_af__activationKernel<<<blocks, threads>>>(
        tensor,
        biases,
        samplesAmount, tensorSize,

        multiplier
    );
}