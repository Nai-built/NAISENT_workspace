#include "../../adjustments.cuh"

#include <iostream>

__global__ void multi_head_KVQ_input__gradientKernel(const kernel_float* __restrict__ keys_gradient, const kernel_float* __restrict__ values_gradient, const kernel_float* __restrict__ queries_gradient
    , const kernel_float* __restrict__ weights, kernel_float* __restrict__ input_gradient
    
    , const int samplesAmount, const int headsAmount, const int inputSize, const int headOutputSize)
{
    __shared__ kernel_float tiledWeights [TILE][TILE][4];

    __shared__ kernel_float tiledKeys_gradient [TILE][TILE];
    __shared__ kernel_float tiledValues_gradient [TILE][TILE];
    __shared__ kernel_float tiledQueries_gradient [TILE][TILE];


    int y = blockIdx.y*TILE + threadIdx.y; // current sample unit
    int x = blockIdx.x*TILE + threadIdx.x; // current input value

    int threadX = threadIdx.x; // does NOT necessarily represent a specific "neuron" location
    int threadY = threadIdx.y; // does NOT necessarily represent batching or a specific sample index

    kernel_float i_g = 0.0f;
    
    for (int h = 0; h < headsAmount; ++h) {
        for (int outputTiling = 0; outputTiling < headOutputSize; outputTiling += TILE) {
            // get cache from main memory to block shared memory
            if (outputTiling+threadY < headOutputSize && x < inputSize) {
                const int indexing = x*headsAmount*headOutputSize*3 + h*headOutputSize*3 + (outputTiling+threadY)*3;
                
                tiledWeights[threadY][threadX][0] = weights[indexing+0];
                tiledWeights[threadY][threadX][1] = weights[indexing+1];
                tiledWeights[threadY][threadX][2] = weights[indexing+2];
            } else {
                tiledWeights[threadY][threadX][0] = 0.0f;
                tiledWeights[threadY][threadX][1] = 0.0f;
                tiledWeights[threadY][threadX][2] = 0.0f;
            }
            if (y < samplesAmount &&  outputTiling+threadX < headOutputSize) {
                const int indexing = y*headsAmount*headOutputSize + h*headOutputSize + (outputTiling+threadX);
                
                tiledKeys_gradient[threadY][threadX] = keys_gradient[indexing];
                tiledValues_gradient[threadY][threadX] = values_gradient[indexing];
                tiledQueries_gradient[threadY][threadX] = queries_gradient[indexing];
            } else {
                tiledKeys_gradient[threadY][threadX] = 0.0f;
                tiledValues_gradient[threadY][threadX] = 0.0f;
                tiledQueries_gradient[threadY][threadX] = 0.0f;
            }
            
            __syncthreads();

            // #pragma unroll TILE
            for (int i = 0; i < TILE; ++i)
            {
                i_g += tiledWeights[i][threadX][0] * tiledKeys_gradient[threadY][i];
                i_g += tiledWeights[i][threadX][1] * tiledValues_gradient[threadY][i];
                i_g += tiledWeights[i][threadX][2] * tiledQueries_gradient[threadY][i];
            }
            __syncthreads();
        }
    }

    if (x < inputSize && y < samplesAmount) input_gradient[y*inputSize + x] = i_g;
}

extern "C" void multi_head_KVQ_input__gradient(const kernel_float* keys_gradient, const kernel_float* values_gradient, const kernel_float* queries_gradient
    , const kernel_float* weights, kernel_float* input_gradient
    , const int& samplesAmount, const int& headsAmount, const int& inputSize, const int& headOutputSize)
{
    dim3 threads(TILE, TILE);
    dim3 blocks(
        (inputSize + TILE - 1) / TILE,
        (samplesAmount + TILE - 1) / TILE
    );
    // launch kernel
    multi_head_KVQ_input__gradientKernel<<<blocks, threads>>>(
        keys_gradient, values_gradient, queries_gradient,
        weights, input_gradient,
        samplesAmount, headsAmount, inputSize, headOutputSize
    );
}