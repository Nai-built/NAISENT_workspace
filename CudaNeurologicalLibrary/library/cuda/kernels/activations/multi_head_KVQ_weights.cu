#include "../../activations.cuh"

#include <iostream>

__global__ void multi_head_KVQ_weights__activationKernel(const kernel_float* __restrict__ input
    , kernel_float* __restrict__ keys, kernel_float* __restrict__ values, kernel_float* __restrict__ queries
    , const kernel_float* __restrict__ weights

    , const int samplesAmount, const int headsAmount, const int inputSize, const int headOutputSize)
{
    __shared__ kernel_float tiledInput [TILE][TILE];
    __shared__ kernel_float tiledWeights [TILE][TILE][TILE][4];
    
    int z = blockIdx.z*TILE + threadIdx.z; // current sample unit
    int y = blockIdx.y*TILE + threadIdx.y; // current head
    int x = blockIdx.x*TILE + threadIdx.x; // current head value

    int threadX = threadIdx.x; // does NOT necessarily represent a specific "neuron" location
    int threadY = threadIdx.y; // does NOT necessarily represent heads or a specific head index
    int threadZ = threadIdx.z; // does NOT necessarily represent batching or a specific sample index

    kernel_float key_result = 0.0f;
    kernel_float value_result = 0.0f;
    kernel_float query_result = 0.0f;

    for (int inputTiling = 0; inputTiling < inputSize; inputTiling += TILE) {
        if (z < samplesAmount && inputTiling+threadX < inputSize) {
            if (threadY == 0) {
                tiledInput[threadZ][threadX] = input[z*inputSize + inputTiling+threadX];
            }
        } else tiledInput[threadZ][threadX] = 0.0f;
        if (x < headOutputSize && y < headsAmount && inputTiling+threadZ < inputSize) {
            const int indexing = (inputTiling+threadZ)*headsAmount*headOutputSize*3 + y*headOutputSize*3 + x*3;

            tiledWeights[threadZ][threadY][threadX][0] = weights[indexing+0];
            tiledWeights[threadZ][threadY][threadX][1] = weights[indexing+1];
            tiledWeights[threadZ][threadY][threadX][2] = weights[indexing+2];
        } else {
            tiledWeights[threadZ][threadY][threadX][0] = 0.0f;
            tiledWeights[threadZ][threadY][threadX][1] = 0.0f;
            tiledWeights[threadZ][threadY][threadX][2] = 0.0f;
        }
        __syncthreads();

        // #pragma unroll TILE
        for (int i = 0; i < TILE; ++i) // compute input * weight
        {
            const kernel_float input_value = tiledInput[threadZ][i];

            key_result += input_value * tiledWeights[i][threadY][threadX][0];
            value_result += input_value * tiledWeights[i][threadY][threadX][1];
            query_result += input_value * tiledWeights[i][threadY][threadX][2];
        }
        __syncthreads();
    }

    if (x < headOutputSize && y < headsAmount && z < samplesAmount) {
        const int indexing = z*headsAmount*headOutputSize + y*headOutputSize + x;
        
        keys[indexing] = key_result;
        values[indexing] = value_result;
        queries[indexing] = query_result;
    }
}

extern "C" void multi_head_KVQ_weights__activation(const kernel_float* input
    , kernel_float* keys, kernel_float* values, kernel_float* queries
    , const kernel_float* weights
    , const int& samplesAmount, const int& headsAmount, const int& inputSize, const int& headOutputSize)
{
    dim3 threads(TILE, TILE, TILE);
    dim3 blocks(
        (headOutputSize + TILE - 1) / TILE,
        (headsAmount + TILE - 1) / TILE,
        (samplesAmount + TILE - 1) / TILE
    );
    // launch kernel
    multi_head_KVQ_weights__activationKernel<<<blocks, threads>>>(
        input,
        keys, values, queries,
        weights,
        samplesAmount, headsAmount, inputSize, headOutputSize
    );
}