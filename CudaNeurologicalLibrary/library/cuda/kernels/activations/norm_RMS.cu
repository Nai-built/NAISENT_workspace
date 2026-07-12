#include "../../activations.cuh"

#include <iostream>

// IMPORTANT NOTE :-
// do NOT try to do "tiling" inside an if statement or anything like it
// let it be free in the actual kernel's function

#define RMSNORM_EPSILON 1e-6f

__global__ void norm_RMS__activationKernel(const kernel_float* __restrict__ input_tensor
    , kernel_float* __restrict__ output_tensor, kernel_float* __restrict__ cache_rMultipliers
    , const kernel_float* __restrict__ gamma

    , const int samplesAmount, const int tensorSize)
{
    __shared__ kernel_float tiledGamma [TILE];

    const int threadX = threadIdx.x;

    int x = blockIdx.x*TILE + threadX; // sample position
    
    kernel_float mean_square = 0.0f;

    if (x < samplesAmount) {
        for (int y = 0; y < tensorSize; ++y) {
            const kernel_float i_v = input_tensor[x*tensorSize + y];
            mean_square += i_v*i_v;
        }
    }

    mean_square /= tensorSize;

    const kernel_float rMultiplier = 1.0f/sqrtf(mean_square+RMSNORM_EPSILON);

    for (int tensorTiling = 0; tensorTiling < tensorSize; tensorTiling += TILE) {
        if (tensorTiling + threadX < tensorSize) {
            // if (gamma[tensorTiling+threadX] == 0.0f) {
            //     printf("ALERT!\n");
            // }
            tiledGamma[threadX] = gamma[tensorTiling+threadX];
        } else tiledGamma[threadX] = 0.0f;
        __syncthreads();

        if (x < samplesAmount) {
            for (int i = 0; i < min(TILE, tensorSize-tensorTiling); ++i) {
                // printf("o: %f", output_tensor[x*tensorSize + tensorTiling+i]);
                // printf("gamma: %f; t: %i; f: %i; tS: %i; tT: %i\n", gamma[i], TILE, min(TILE, tensorSize-tensorTiling), tensorSize, tensorTiling);
                output_tensor[x*tensorSize + tensorTiling+i]
                = input_tensor[x*tensorSize + tensorTiling+i]*rMultiplier * tiledGamma[i];
                // printf("tttt\n");
            }
        }
        __syncthreads();
    }
    
    if (x < samplesAmount) {
        cache_rMultipliers[x] = rMultiplier;
    }
}

extern "C" void norm_RMS__activation(const kernel_float* input_tensor
    , kernel_float* output_tensor, kernel_float* cache_rMultipliers
    , const kernel_float* gamma
    , const int& samplesAmount, const int& tensorSize)
{
    dim3 threads(TILE);
    dim3 blocks(
        (samplesAmount + TILE - 1) / TILE
    );
    // launch kernel
    norm_RMS__activationKernel<<<blocks, threads>>>(
        input_tensor,
        output_tensor, cache_rMultipliers,
        gamma,
        samplesAmount, tensorSize
    );
}