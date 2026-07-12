#include "../../adjustments.cuh"

#include <iostream>

__global__ void input_RMS__gradientKernel(const kernel_float* __restrict__ output_gradient
    , const kernel_float* __restrict__ cached_input, const kernel_float* __restrict__ cached_rMultipliers
    , const kernel_float* __restrict__ gamma
    , kernel_float* __restrict__ input_gradient

    , const int samplesAmount, const int tensorSize)
{
    __shared__ kernel_float tiledGamma [TILE];

    const int threadX = threadIdx.x;

    int x = blockIdx.x*TILE + threadX; // sample position
    
    const kernel_float rMultiplier = x < samplesAmount ? cached_rMultipliers[x] : 0.0f;
    const kernel_float r = x < samplesAmount ? 1.0f/rMultiplier : 0.0f;
    const kernel_float squaredRMultiplier = x < samplesAmount ? 1.0f/(tensorSize*r*r*r) : 0.0f;

    kernel_float dot = 0.0f;

    // compute dot multiplier
    for (int tensorTiling = 0; tensorTiling < tensorSize; tensorTiling += TILE) {
        if (tensorTiling + threadX < tensorSize) {
            tiledGamma[threadX] = gamma[tensorTiling+threadX];
        } else tiledGamma[threadX] = 0.0f;
        __syncthreads();

        if (x < samplesAmount) {
            for (int i = 0; i < min(TILE, tensorSize-tensorTiling); ++i) {
                dot += output_gradient[x*tensorSize + tensorTiling+i]*rMultiplier * tiledGamma[i];
            }
        }
        __syncthreads();
    }
        
    // process gradient
    for (int tensorTiling = 0; tensorTiling < tensorSize; tensorTiling += TILE) {
        if (tensorTiling + threadX < tensorSize) {
            tiledGamma[threadX] = gamma[tensorTiling+threadX];
        } else tiledGamma[threadX] = 0.0f;
        __syncthreads();

        if (x < samplesAmount) {
            for (int i = 0; i < min(TILE, tensorSize-tensorTiling); ++i) {
                input_gradient[x*tensorSize + tensorTiling+i]
                = (output_gradient[x*tensorSize + tensorTiling+i]*rMultiplier * tiledGamma[i])
                - (cached_input[x*tensorSize + tensorTiling+i]*dot * squaredRMultiplier);
            }
        }
        __syncthreads();
    }
}

extern "C" void input_RMS__gradient(const kernel_float* output_gradient
    , const kernel_float* cached_input, const kernel_float* cached_rMultipliers
    , const kernel_float* gamma
    , kernel_float* input_gradient
    , const int& samplesAmount, const int& tensorSize)
{
    dim3 threads(TILE);
    dim3 blocks(
        (samplesAmount + TILE - 1) / TILE
    );
    // launch kernel
    input_RMS__gradientKernel<<<blocks, threads>>>(
        output_gradient,
        cached_input, cached_rMultipliers,
        gamma,
        input_gradient,
        samplesAmount, tensorSize
    );
}