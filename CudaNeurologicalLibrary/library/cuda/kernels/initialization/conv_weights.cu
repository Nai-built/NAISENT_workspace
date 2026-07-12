#include "../../initialization.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <iostream>

// FROM ChatGPT
__global__ void conv__rng_setup_kernel(
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

__global__ void conv_weights__initializationKernel(const int inputChannels, const int outputChannels
    , const int slideWidth, const int slideHeight
    , kernel_float* __restrict__ weights
    
    , curandStatePhilox4_32_10_t* states
    , kernel_float rand_multiplier)
{
    int x = blockIdx.x*TILE + threadIdx.x; // output position
    
    curandStatePhilox4_32_10_t state = states[x];

    if (x < outputChannels) {
        for (int y = 0; y < inputChannels; ++y) {
            for (int i = 0; i < slideWidth*slideHeight; ++i) {
                weights[y*outputChannels*slideWidth*slideHeight + x*slideWidth*slideHeight + i] = (curand_uniform(&state)*2 - 1)*rand_multiplier * 1;
            }
        }
    }
}

extern "C" void conv_weights__initialization(const int& inputChannels, const int& outputChannels
    , const int& slideWidth, const int& slideHeight
    , kernel_float* weights

    , unsigned long initializationSeed)
{
    dim3 threads(TILE);
    dim3 blocks(
        (outputChannels + TILE - 1) / TILE
    );
    int totalThreads = threads.x * blocks.x;
    // launch kernel
    curandStatePhilox4_32_10_t* rand_states = nullptr;
    cudaMalloc(&rand_states,
            totalThreads * sizeof(curandStatePhilox4_32_10_t));
    conv__rng_setup_kernel<<<blocks, threads>>>(rand_states, initializationSeed, totalThreads);
    cudaDeviceSynchronize();

	kernel_float weights_initialization_boundary =
    (sqrt(2.0 / ((double)(inputChannels)))*sqrt(3));
    conv_weights__initializationKernel<<<blocks, threads>>>(
        inputChannels, outputChannels,
        slideWidth, slideHeight,
        weights,

        rand_states, weights_initialization_boundary
    );
    cudaDeviceSynchronize();
    cudaFree(rand_states);
}