#include "../../activations.cuh"

#include <iostream>

__global__ void SCC_positional_embedding__activationKernel(kernel_float* __restrict__ output
    , const kernel_float* __restrict__ sine_multipliers
    , const kernel_float* __restrict__ cosine_multipliers

    , const int length
    , const int kernel_sample_start
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
            output[(kernel_sample_start+y)*embeddingSize + x] += cos(tiledMultipliers[threadX]*y);
        } else {
            // sine
            output[(kernel_sample_start+y)*embeddingSize + x] += sin(tiledMultipliers[threadX]*y);
        }
    }
}

extern "C" void SCC_positional_embedding__activation(kernel_float* output
    , const kernel_float* sine_multipliers
    , const kernel_float* cosine_multipliers

    , const int& batchSize
    , const int* seriesLengths
    , const int& embeddingSize)
{
    int sample_start = 0;

    for (int b = 0; b < batchSize; ++b) {
        const int l = seriesLengths[b];

        dim3 threads(TILE, TILE);
        dim3 blocks((embeddingSize + TILE - 1) / TILE, (l + TILE - 1) / TILE);
        SCC_positional_embedding__activationKernel<<<blocks, threads>>>(
            output,
            sine_multipliers, cosine_multipliers,
            l, sample_start,
            embeddingSize
        );

        sample_start += l;
    }
}