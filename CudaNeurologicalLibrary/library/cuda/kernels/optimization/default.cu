#include "../../optimization.cuh"

__global__ void default__applyKernel(const int size
    , kernel_float* __restrict__ parameters
    , const kernel_float* __restrict__ derivatives

    , const kernel_float multiplier)
{
    int x = blockIdx.x*TILE + threadIdx.x; // output position
    
    if (x < size) {
        parameters[x] -= derivatives[x] * multiplier;
    }
}

extern "C" void default__apply(const int& size
    , kernel_float* parameters
    , const kernel_float* derivatives
    , const kernel_float& multiplier)
{
    dim3 threads(TILE);
    dim3 blocks(
        (size + TILE - 1) / TILE
    );
    // launch kernel
    default__applyKernel<<<blocks, threads>>>(
        size,
        parameters,
        derivatives,
        multiplier
    );
}