#include "../../adjustments.cuh"

#include <iostream>

// __global__ void fc_weights_gradientKernel(
//     const kernel_float* output_gradient,
//     const kernel_float* cached_input,
//     kernel_float* weights_gradient,
//     const int samplesAmount,
//     const int inputSize,
//     const int outputSize)
// {
//     __shared__ kernel_float sharedInput[TILE][TILE];

//     int bx = blockIdx.x * TILE; // input tile start
//     int by = blockIdx.y * TILE; // output tile start

//     int tx = threadIdx.x;
//     int ty = threadIdx.y;

//     // local accumulation for each thread
//     kernel_float local_acc = 0.0f;

//     for (int s = 0; s < samplesAmount; ++s)
//     {
//         // load input tile
//         if (bx + tx < inputSize)
//             sharedInput[ty][tx] = cached_input[s * inputSize + bx + tx];
//         else
//             sharedInput[ty][tx] = 0.0f;

//         __syncthreads();

//         // load output gradient for this sample
//         kernel_float g = (by + ty < outputSize) ? output_gradient[s * outputSize + by + ty] : 0.0f;

//         // compute local contributions
// #pragma unroll
//         for (int i = 0; i < TILE; ++i)
//             local_acc += sharedInput[tx][i] * g;

//         __syncthreads();
//     }

//     // write accumulated tile once
//     if (bx + tx < inputSize && by + ty < outputSize)
//         atomicAdd(&weights_gradient[(bx + tx) * outputSize + by + ty], local_acc);
// }

// __global__ void fc_weights_gradientKernel(
//     const kernel_float* output_gradient,
//     const kernel_float* cached_input,
//     kernel_float* weights_gradient,
//     const int samplesAmount,
//     const int inputSize,
//     const int outputSize)
// {
//     __shared__ kernel_float tiledCached_input[TILE][TILE];

//     int y = blockIdx.y * TILE + threadIdx.y; // sample index
//     int x = blockIdx.x * TILE + threadIdx.x; // output neuron index

//     int threadX = threadIdx.x;
//     int threadY = threadIdx.y;

//     // get output gradient once
//     const kernel_float g = (x < outputSize && y < samplesAmount) ? output_gradient[y * outputSize + x] : 0.0f;

//     for (int inputTiling = 0; inputTiling < inputSize; inputTiling += TILE)
//     {
//         // load cached input tile
//         if (y < samplesAmount && inputTiling + threadX < inputSize)
//             tiledCached_input[threadY][threadX] = cached_input[y * inputSize + inputTiling + threadX];
//         else
//             tiledCached_input[threadY][threadX] = 0.0f;

//         __syncthreads();

//         // accumulate weights gradient in register
//         kernel_float w_g = 0.0f;
// #pragma unroll
//         for (int i = 0; i < TILE; ++i)
//             w_g += tiledCached_input[threadY][i] * g;

//         // atomic add exactly once per thread per tile
//         if (x < outputSize && inputTiling + threadY < inputSize)
//             atomicAdd(&weights_gradient[(inputTiling + threadY) * outputSize + x], w_g);

//         __syncthreads(); // sync before next tile
//     }
// }

__global__ void fc_weights__gradientKernel(const kernel_float* __restrict__ output_gradient
    , const kernel_float* __restrict__ cached_input, kernel_float* __restrict__ weights_gradient
    
    , const int samplesAmount, const int inputSize, const int outputSize)
{
    __shared__ kernel_float tiledCached_input [TILE][TILE];
    __shared__ kernel_float tiledOutput_gradient [TILE][TILE];

    int y = blockIdx.y*TILE + threadIdx.y; // current input
    int x = blockIdx.x*TILE + threadIdx.x; // current output

    int threadX = threadIdx.x; // does NOT necessarily represent a specific "neuron" location
    int threadY = threadIdx.y; // does NOT necessarily represent batching or a specific sample index

    kernel_float w_g = 0.0f;
    
    for (int samplesTiling = 0; samplesTiling < samplesAmount; samplesTiling += TILE) {
        // get cached input from main memory to block shared memory
        if (samplesTiling+threadX < samplesAmount && y < inputSize) {
            tiledCached_input[threadY][threadX] = cached_input[(samplesTiling+threadX)*inputSize + y];
        } else tiledCached_input[threadY][threadX] = 0.0f;
        if (x < outputSize &&  samplesTiling+threadY < samplesAmount) {
            tiledOutput_gradient[threadY][threadX] = output_gradient[(samplesTiling+threadY)*outputSize + x];
        } else tiledOutput_gradient[threadY][threadX] = 0.0f;
        
        __syncthreads();

        // #pragma unroll TILE
        for (int i = 0; i < TILE; ++i)
        {
            const kernel_float v = tiledCached_input[threadY][i] * tiledOutput_gradient[i][threadX];
            w_g += v;
        }
        __syncthreads();
    }

    if (x < outputSize && y < inputSize) weights_gradient[y*outputSize + x] += w_g;
}

extern "C" void fc_weights__gradient(const kernel_float* output_gradient
    , const kernel_float* cached_input, kernel_float* weights_gradient
    , const int& samplesAmount, const int& inputSize, const int& outputSize)
{
    dim3 threads(TILE, TILE);
    dim3 blocks(
        (outputSize + TILE - 1) / TILE,
        (inputSize + TILE - 1) / TILE
    );
    // launch kernel
    fc_weights__gradientKernel<<<blocks, threads>>>(
        output_gradient,
        cached_input, weights_gradient,
        samplesAmount, inputSize, outputSize
    );
}