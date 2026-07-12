#include "../../initialization.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <iostream>

__global__ void multi_head_KVQ__rng_setup_kernel(
    curandStatePhilox4_32_10_t* __restrict__ states,
    unsigned long seed,
    int totalThreads
)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id >= totalThreads) return;

    curand_init(
        seed,
        id,
        0,
        &states[id]
    );
}

__global__ void multi_head_KVQ__initializationKernel(const int inputSize, const int totalOutputSize
    , kernel_float* __restrict__ weights

    , curandStatePhilox4_32_10_t* states
    , kernel_float rand_multiplier)
{
    int x = blockIdx.x*TILE + threadIdx.x; // output position (across headsAmount * headOutputSize * 3)

    curandStatePhilox4_32_10_t state = states[x];

    if (x < totalOutputSize) {
        for (int y = 0; y < inputSize; ++y) {
            weights[y*totalOutputSize + x] = (curand_uniform(&state)*2 - 1)*rand_multiplier;
        }
    }
}

extern "C" void multi_head_KVQ__initialization(const int& inputSize, const int& headsAmount, const int& headOutputSize
    , kernel_float* weights

    , unsigned long initializationSeed)
{
    const int totalOutputSize = headsAmount * headOutputSize * 3;

    dim3 threads(TILE);
    dim3 blocks(
        (totalOutputSize + TILE - 1) / TILE
    );
    int totalThreads = threads.x * blocks.x;

    curandStatePhilox4_32_10_t* rand_states = nullptr;
    cudaMalloc(&rand_states,
            totalThreads * sizeof(curandStatePhilox4_32_10_t));
    multi_head_KVQ__rng_setup_kernel<<<blocks, threads>>>(rand_states, initializationSeed, totalThreads);
    cudaDeviceSynchronize();

    kernel_float weights_initialization_boundary =
        (sqrt(2.0 / ((double)(inputSize)))*sqrt(3));
    multi_head_KVQ__initializationKernel<<<blocks, threads>>>(
        inputSize, totalOutputSize,
        weights,

        rand_states, weights_initialization_boundary
    );
    cudaDeviceSynchronize();
    cudaFree(rand_states);
}
