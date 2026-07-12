#include "../../adjustments.cuh"

#include <iostream>

__global__ void fc_input__gradientKernel(const kernel_float* __restrict__ output_gradient
    , const kernel_float* __restrict__ weights, kernel_float* __restrict__ input_gradient
    
    , const int samplesAmount, const int inputSize, const int outputSize)
{
    __shared__ kernel_float tiledWeights [TILE][TILE];
    __shared__ kernel_float tiledOutput_gradient [TILE][TILE];

    int y = blockIdx.y*TILE + threadIdx.y; // current sample unit
    int x = blockIdx.x*TILE + threadIdx.x; // current sample input value

    int threadX = threadIdx.x; // does NOT necessarily represent a specific "neuron" location
    int threadY = threadIdx.y; // does NOT necessarily represent batching or a specific sample index

    kernel_float i_g = 0.0f;
    // kernel_float c = 0.0f;
    
    for (int outputTiling = 0; outputTiling < outputSize; outputTiling += TILE) {
        // get cache from main memory to block shared memory
        if (outputTiling+threadY < outputSize && x < inputSize) {
            tiledWeights[threadY][threadX] = weights[x*outputSize + outputTiling+threadY];
        } else tiledWeights[threadY][threadX] = 0.0f;
        if (y < samplesAmount &&  outputTiling+threadX < outputSize) {
            tiledOutput_gradient[threadY][threadX] = output_gradient[y*outputSize + outputTiling+threadX];
        } else tiledOutput_gradient[threadY][threadX] = 0.0f;
        
        __syncthreads();

        // #pragma unroll TILE
        for (int i = 0; i < TILE; ++i)
        {
            const kernel_float v = tiledWeights[i][threadX] * tiledOutput_gradient[threadY][i];
            i_g += v;
        }
        __syncthreads();
    }

    if (x < inputSize && y < samplesAmount) input_gradient[y*inputSize + x] = i_g;
}

extern "C" void fc_input__gradient(const kernel_float* output_gradient
    , const kernel_float* weights, kernel_float* input_gradient
    , const int& samplesAmount, const int& inputSize, const int& outputSize)
{
    dim3 threads(TILE, TILE);
    dim3 blocks(
        (inputSize + TILE - 1) / TILE,
        (samplesAmount + TILE - 1) / TILE
    );
    // launch kernel
    fc_input__gradientKernel<<<blocks, threads>>>(
        output_gradient,
        weights, input_gradient,
        samplesAmount, inputSize, outputSize
    );
}