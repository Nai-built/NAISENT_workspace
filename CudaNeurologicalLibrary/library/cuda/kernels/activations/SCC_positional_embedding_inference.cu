#include "../../activations.cuh"

#include <iostream>

__global__ void SCC_positional_embedding_inference__activationKernel(kernel_float* __restrict__ output
    , const kernel_float* __restrict__ sine_multipliers
    , const kernel_float* __restrict__ cosine_multipliers

    , const int pre_length
    , const int length
    , const int embeddingSize)
{
    __shared__ kernel_float tiledMultipliers [TILE];
    
    int y = blockIdx.y*TILE + threadIdx.y; // current sample unit
    int x = blockIdx.x*TILE + threadIdx.x; // current sample value

    int threadX = threadIdx.x; // does NOT necessarily represent a specific "neuron" location
    int threadY = threadIdx.y; // does NOT necessarily represent batching or a specific sample index

    if (threadY == 0) {
        if (x < embeddingSize) {
            if (x % 2 == 0) {
                // cosine
                tiledMultipliers[threadX] = cosine_multipliers[x/2];
            } else {
                // sine
                tiledMultipliers[threadX] = sine_multipliers[(x+1)/2];
            }
        } else tiledMultipliers[threadX] = 0.0f;
    }
    __syncthreads();

    if (x < embeddingSize && y < length) {
        if (x % 2 == 0) {
            // cosine
            output[(y)*embeddingSize + x] += cos(tiledMultipliers[threadX]*(y+pre_length));
        } else {
            // sine
            output[(y)*embeddingSize + x] += sin(tiledMultipliers[threadX]*(y+pre_length));
        }
    }
}

extern "C" void SCC_positional_embedding_inference__activation(kernel_float* output
    , const kernel_float* sine_multipliers
    , const kernel_float* cosine_multipliers

    , const int& pre_length
    , const int& length
    , const int& embeddingSize)
{
    dim3 threads(TILE, TILE);
    dim3 blocks((embeddingSize + TILE - 1) / TILE, (length + TILE - 1) / TILE);
    SCC_positional_embedding_inference__activationKernel<<<blocks, threads>>>(
        output,
        sine_multipliers, cosine_multipliers,
        pre_length, length,
        embeddingSize
    );
}