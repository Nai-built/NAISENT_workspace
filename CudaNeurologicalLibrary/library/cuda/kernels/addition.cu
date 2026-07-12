#include "../cuda_base.cuh"

#include <iostream>

__global__ void additionKernel(kernel_float* __restrict__ main, const kernel_float* __restrict__ other
    , const int mainSize, const int otherSize)
{
    const int x = blockIdx.x*TILE + threadIdx.x;

    if (x < mainSize && x < otherSize) {
        main[x] += other[x];
    }
}

extern "C" void addition(kernel_float* main, const kernel_float* other
    , const int& mainSize, const int& otherSize)
{
    dim3 threads(TILE);
    dim3 blocks((mainSize + TILE - 1) / TILE);

    additionKernel<<<blocks, threads>>>(main, other, mainSize, otherSize);
}