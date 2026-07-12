#include "../../adjustments.cuh"

#include <iostream>

__global__ void multi_head_KVQ_weights__gradientKernel(const kernel_float* __restrict__ keys_gradient, const kernel_float* __restrict__ values_gradient, const kernel_float* __restrict__ queries_gradient
    , const kernel_float* __restrict__ cached_input, kernel_float* __restrict__ weights_gradient
    
    , const int samplesAmount, const int headsAmount, const int inputSize, const int headOutputSize)
{
    __shared__ kernel_float tiledCached_input [TILE][TILE];

    __shared__ kernel_float tiledKeys_gradient [TILE][TILE][TILE];
    __shared__ kernel_float tiledValues_gradient [TILE][TILE][TILE];
    __shared__ kernel_float tiledQueries_gradient [TILE][TILE][TILE];

    int z = blockIdx.z*TILE + threadIdx.z; // current input
    int y = blockIdx.y*TILE + threadIdx.y; // current head
    int x = blockIdx.x*TILE + threadIdx.x; // current head output

    int threadX = threadIdx.x; // does NOT necessarily represent a specific "neuron" location
    int threadY = threadIdx.y; // does NOT necessarily represent heads or a specific head index
    int threadZ = threadIdx.z; // does NOT necessarily represent batching or a specific sample index

    kernel_float key_w_g = 0.0f;
    kernel_float value_w_g = 0.0f;
    kernel_float query_w_g = 0.0f;
    
    for (int samplesTiling = 0; samplesTiling < samplesAmount; samplesTiling += TILE) {
        // get cached input from main memory to block shared memory
        if (samplesTiling+threadX < samplesAmount && z < inputSize) {
            if (threadY == 0) {
                tiledCached_input[threadZ][threadX] = cached_input[(samplesTiling+threadX)*inputSize + z];
            }
        } else tiledCached_input[threadZ][threadX] = 0.0f;
        if (x < headOutputSize && y < headsAmount &&  samplesTiling+threadZ < samplesAmount) {
            const int indexing = (samplesTiling+threadZ)*headsAmount*headOutputSize + y*headOutputSize + x;

            tiledKeys_gradient[threadZ][threadY][threadX] = keys_gradient[indexing];
            tiledValues_gradient[threadZ][threadY][threadX] = values_gradient[indexing];
            tiledQueries_gradient[threadZ][threadY][threadX] = queries_gradient[indexing];
        } else {
            tiledKeys_gradient[threadZ][threadY][threadX] = 0.0f;
            tiledValues_gradient[threadZ][threadY][threadX] = 0.0f;
            tiledQueries_gradient[threadZ][threadY][threadX] = 0.0f;
        }
        
        __syncthreads();

        // #pragma unroll TILE
        for (int i = 0; i < TILE; ++i)
        {
            const kernel_float cached_input_value = tiledCached_input[threadZ][i];

            key_w_g += cached_input_value * tiledKeys_gradient[i][threadY][threadX];
            value_w_g += cached_input_value * tiledValues_gradient[i][threadY][threadX];
            query_w_g += cached_input_value * tiledQueries_gradient[i][threadY][threadX];
        }
        __syncthreads();
    }

    if (x < headOutputSize && y < headsAmount && z < inputSize) {
        const int indexing = z*headsAmount*headOutputSize*3 + y*headOutputSize*3 + x*3;

        weights_gradient[indexing+0] += key_w_g;
        weights_gradient[indexing+1] += value_w_g;
        weights_gradient[indexing+2] += query_w_g;
    }
}

extern "C" void multi_head_KVQ_weights__gradient(const kernel_float* keys_gradient, const kernel_float* values_gradient, const kernel_float* queries_gradient
    , const kernel_float* cached_input, kernel_float* weights_gradient
    , const int& samplesAmount, const int& headsAmount, const int& inputSize, const int& headOutputSize)
{
    dim3 threads(TILE, TILE, TILE);
    dim3 blocks(
        (headOutputSize + TILE - 1) / TILE,
        (headsAmount + TILE - 1) / TILE,
        (inputSize + TILE - 1) / TILE
    );
    // launch kernel
    multi_head_KVQ_weights__gradientKernel<<<blocks, threads>>>(
        keys_gradient, values_gradient, queries_gradient,
        cached_input, weights_gradient,
        samplesAmount, headsAmount, inputSize, headOutputSize
    );
}