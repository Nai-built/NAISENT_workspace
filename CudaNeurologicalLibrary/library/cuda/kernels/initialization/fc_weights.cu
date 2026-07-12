#include "../../initialization.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <iostream>

// FROM ChatGPT
__global__ void rng_setup_kernel(
    curandStatePhilox4_32_10_t* __restrict__ states,
    unsigned long seed,
    int totalThreads
)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id >= totalThreads) return;

    curand_init(
        seed,   // global seed
        id,     // unique sequence per thread
        0,      // offset
        &states[id]
    );
}

__global__ void fc_weights__initializationKernel(const int inputSize, const int outputSize
    , kernel_float* __restrict__ weights
    
    , curandStatePhilox4_32_10_t* states
    , kernel_float rand_multiplier)
{
    int x = blockIdx.x*TILE + threadIdx.x; // output position
    
    curandStatePhilox4_32_10_t state = states[x];

    if (x < outputSize) {
        for (int y = 0; y < inputSize; ++y) {
            weights[y*outputSize + x] = (curand_uniform(&state)*2 - 1)*rand_multiplier * 1;
            // printf("x %d: y %d: weight = %f\n", x, y, weights[y*outputSize + x]);
        }
    }
}

extern "C" void fc_weights__initialization(const int& inputSize, const int& outputSize
    , kernel_float* weights

    , unsigned long initializationSeed)
{
    dim3 threads(TILE);
    dim3 blocks(
        (outputSize + TILE - 1) / TILE
    );
    int totalThreads = threads.x * blocks.x;
    // launch kernel
    curandStatePhilox4_32_10_t* rand_states = nullptr;
    cudaMalloc(&rand_states,
            totalThreads * sizeof(curandStatePhilox4_32_10_t));
    rng_setup_kernel<<<blocks, threads>>>(rand_states, initializationSeed, totalThreads);
    cudaDeviceSynchronize();

	kernel_float weights_initialization_boundary =
    (sqrt(2.0 / ((double)(inputSize)))*sqrt(3));
    fc_weights__initializationKernel<<<blocks, threads>>>(
        inputSize, outputSize,
        weights,

        rand_states, weights_initialization_boundary
    );
    cudaDeviceSynchronize();
    cudaFree(rand_states);
}