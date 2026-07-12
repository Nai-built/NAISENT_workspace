#include "../../activations.cuh"

#include <iostream>

__global__ void linear_dense__activationKernel(const kernel_float* input, kernel_float* output
    , const kernel_float* weights, const kernel_float* biases
    
    , const int samplesAmount, const int inputSize, const int outputSize)
{
    __shared__ kernel_float tiledInput [TILE][TILE];
    __shared__ kernel_float tiledWeights [TILE][TILE];
    
    __shared__ kernel_float tiledBiases [TILE];

    int y = blockIdx.y*TILE + threadIdx.y; // current sample unit
    int x = blockIdx.x*TILE + threadIdx.x; // current sample value

    int threadX = threadIdx.x; // does NOT necessarily represent a specific "neuron" location
    int threadY = threadIdx.y; // does NOT necessarily represent batching or a specific sample index

    kernel_float result = 0.0f;

    if (threadY == 0) {
        if (x < outputSize) {
            tiledBiases[threadX] = biases[x];
        } else tiledBiases[threadX] = 0.0f;
    }
    __syncthreads();

    for (int inputTiling = 0; inputTiling < inputSize; inputTiling += TILE) {
        if (y < samplesAmount && inputTiling+threadX < inputSize) {
            tiledInput[threadY][threadX] = input[y*inputSize + inputTiling+threadX];
        } else tiledInput[threadY][threadX] = 0.0f;
        if (x < outputSize && inputTiling+threadY < inputSize) {
            tiledWeights[threadY][threadX] = weights[(inputTiling+threadY)*outputSize + x];
        } else tiledWeights[threadY][threadX] = 0.0f;
        __syncthreads();

        for (int i = 0; i < TILE; ++i) // compute input * weight
        {
            // printf("Thread %d: weight = %f: input = %f\n", y*outputSize + x, tiledWeights[i][threadX], tiledInput[threadY][i]);
            result += tiledInput[threadY][i] * tiledWeights[i][threadX];
        }
        __syncthreads();
    }

    if (x < outputSize && y < samplesAmount) output[y*outputSize + x] = result + tiledBiases[threadX];
}

extern "C" void linear_dense__activation(const kernel_float* input, kernel_float* output
    , const kernel_float* weights, const kernel_float* biases
    , const int& samplesAmount, const int& inputSize, const int& outputSize)
{
    dim3 threads(TILE, TILE);
    dim3 blocks(
        (outputSize + TILE - 1) / TILE,
        (samplesAmount + TILE - 1) / TILE
    );
    // launch kernel
    linear_dense__activationKernel<<<blocks, threads>>>(
        input, output,
        weights, biases,
        samplesAmount, inputSize, outputSize
    );
    cudaDeviceSynchronize();
    // printf("Thread %d: weight = %f: input = %f\n", y*outputSize + x, tiledWeights[i][threadX], tiledInput[threadY][i]);
}