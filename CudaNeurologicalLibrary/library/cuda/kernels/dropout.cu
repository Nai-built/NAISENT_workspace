#include "../cuda_base.cuh"

#include <iostream>

__global__ void dropoutKernel(kernel_float* __restrict__ tensor
    , const int* __restrict__ dropout_mask
    , const int tensorSize
    , const int dropoutSize, const int samplesAmount)
{
    const int x = blockIdx.x*TILE + threadIdx.x;

    if (x < dropoutSize) {
        const int d = dropout_mask[x];
        // printf("d: %i", d);
        for (int y = 0; y < samplesAmount; ++y) {
            tensor[y*tensorSize + d] = 0.0f;
        }
    }
}

extern "C" void dropout(kernel_float* tensor
    , const int* dropout_mask
    , const int& tensorSize
    , const int& dropoutSize, const int& samplesAmount)
{
    dim3 threads(TILE);
    dim3 blocks(
        (dropoutSize + TILE - 1) / TILE
    );
    // launch kernel
    dropoutKernel<<<blocks, threads>>>(
        tensor,
        dropout_mask,
        tensorSize,
        dropoutSize, samplesAmount
    );
}