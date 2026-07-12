#include "../../adjustments.cuh"

#include <iostream>

#define RMSNORM_EPSILON 1e-6f

__global__ void gamma_RMS__gradientKernel(const kernel_float* __restrict__ output_gradient
    , const kernel_float* __restrict__ cached_input, const kernel_float* __restrict__ cached_rMultipliers
    , kernel_float* __restrict__ gamma_gradient

    , const int samplesAmount, const int tensorSize)
{
    // __shared__ kernel_float tiledInput [TILE];
    // __shared__ kernel_float tiledOutput_gradient [TILE];
    __shared__ kernel_float tiledRMultipliers [TILE];

    const int threadX = threadIdx.x;

    int x = blockIdx.x*TILE + threadX; // tensor position
    
    kernel_float g_g = 0.0f;

    for (int samplesTiling = 0; samplesTiling < samplesAmount; samplesTiling += TILE) {
        if (samplesTiling + threadX < samplesAmount) {
            // tiledInput[threadX] = cached_input[(samplesTiling + threadX)*tensorSize + x];
            // tiledOutput_gradient[threadX] = output_gradient[(samplesTiling + threadX)*tensorSize + x];
            tiledRMultipliers[threadX] = cached_rMultipliers[samplesTiling+threadX];
        } else {
            // tiledInput[threadX] = 0.0f;
            // tiledOutput_gradient[threadX] = 0.0f;
            tiledRMultipliers[threadX] = 0.0f;
        }
        __syncthreads();

        if (x < tensorSize) {
            for (int i = 0; i < min(TILE, samplesAmount-samplesTiling); ++i) {
                // g_g += tiledInput[i]*tiledRMultipliers[i] * tiledOutput_gradient[i];
                g_g += cached_input[(samplesTiling + i)*tensorSize + x]*tiledRMultipliers[i] * output_gradient[(samplesTiling + i)*tensorSize + x];
            }
        }
        __syncthreads();
    }

    if (x < tensorSize) {
        gamma_gradient[x] += g_g;
    }
}

extern "C" void gamma_RMS__gradient(const kernel_float* output_gradient
    , const kernel_float* cached_input, const kernel_float* cached_rMultipliers
    , kernel_float* gamma_gradient
    , const int& samplesAmount, const int& tensorSize)
{
    dim3 threads(TILE);
    dim3 blocks(
        (tensorSize + TILE - 1) / TILE
    );
    // launch kernel
    gamma_RMS__gradientKernel<<<blocks, threads>>>(
        output_gradient,
        cached_input, cached_rMultipliers,
        gamma_gradient,
        samplesAmount, tensorSize
    );
}