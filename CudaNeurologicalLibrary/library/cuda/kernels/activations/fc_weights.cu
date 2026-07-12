#include "../../activations.cuh"

#include <iostream>

            // const int weightPosition = (inputTiling+i)*outputSize + x;
            // const int inputPosition = inputTiling+i;
            // const int outputPosition = x;
            // const int samplePosition = y;
            // if (weightPosition < inputSize*outputSize && inputPosition < inputSize && outputPosition < outputSize && samplePosition < samplesAmount) {
            //     printf("weight: %i; input: %i; output: %i; sample: %i\n"
            //         , weightPosition, inputPosition, outputPosition, samplePosition);
            // }
            // printf("Thread %d: weight = %f: input = %f\n", y*outputSize + x, tiledWeights[i][threadX], tiledInput[threadY][i]);


__global__ void fc_weights__activationKernel(const kernel_float* __restrict__ input, kernel_float* __restrict__ output
    , const kernel_float* __restrict__ weights
    
    , const int samplesAmount, const int inputSize, const int outputSize)
{
    __shared__ kernel_float tiledInput [TILE][TILE];
    __shared__ kernel_float tiledWeights [TILE][TILE];
    
    int y = blockIdx.y*TILE + threadIdx.y; // current sample unit
    int x = blockIdx.x*TILE + threadIdx.x; // current sample value

    int threadX = threadIdx.x; // does NOT necessarily represent a specific "neuron" location
    int threadY = threadIdx.y; // does NOT necessarily represent batching or a specific sample index

    kernel_float result = 0.0f;

    for (int inputTiling = 0; inputTiling < inputSize; inputTiling += TILE) {
        if (y < samplesAmount && inputTiling+threadX < inputSize) {
            tiledInput[threadY][threadX] = input[y*inputSize + inputTiling+threadX];
        } else tiledInput[threadY][threadX] = 0.0f;
        if (x < outputSize && inputTiling+threadY < inputSize) {
            tiledWeights[threadY][threadX] = weights[(inputTiling+threadY)*outputSize + x];
        } else tiledWeights[threadY][threadX] = 0.0f;
        __syncthreads();

        // #pragma unroll TILE
        for (int i = 0; i < TILE; ++i) // compute input * weight
        {
            result += tiledInput[threadY][i] * tiledWeights[i][threadX];
        }
        __syncthreads();
    }

    if (x < outputSize && y < samplesAmount) output[y*outputSize + x] = result;
}

extern "C" void fc_weights__activation(const kernel_float* input, kernel_float* output
    , const kernel_float* weights
    , const int& samplesAmount, const int& inputSize, const int& outputSize)
{
    dim3 threads(TILE, TILE);
    dim3 blocks(
        (outputSize + TILE - 1) / TILE,
        (samplesAmount + TILE - 1) / TILE
    );
    // launch kernel
    fc_weights__activationKernel<<<blocks, threads>>>(
        input, output,
        weights,
        samplesAmount, inputSize, outputSize
    );
}