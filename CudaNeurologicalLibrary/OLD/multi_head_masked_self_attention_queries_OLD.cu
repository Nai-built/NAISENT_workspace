#include "../../adjustments.cuh"

#include <iostream>

__global__ void multi_head_masked_self_attention_queries__gradientKernel(
    const kernel_float* __restrict__ final_output_gradient,
    const kernel_float* __restrict__ keys,
    const kernel_float* __restrict__ values,
    const kernel_float* __restrict__ cache_softmax,
    kernel_float* __restrict__ cache_score_gradients,
    kernel_float* __restrict__ queries_gradient,
    const int batchSize, const int* __restrict__ seriesLengths, const int totalSamplesAmount,
    const int headsAmount, const int inputSize, const int headOutputSize)
{
    int head_dim  = blockIdx.x * TILE + threadIdx.x;
    int head      = blockIdx.y * TILE + threadIdx.y;
    int token_idx = blockIdx.z * TILE + threadIdx.z;

    if (head_dim >= headOutputSize || head >= headsAmount || token_idx >= totalSamplesAmount) return;

    // find batch and local position
    int batch_start = 0;
    int length = 0;
    for (int b = 0; b < batchSize; ++b) {
        int l = seriesLengths[b];
        if (batch_start + l > token_idx) { length = l; break; }
        batch_start += l;
    }
    int pos = token_idx - batch_start;

    const kernel_float scale = rsqrtf((kernel_float)headOutputSize);

    const kernel_float* __restrict__ d_out      = final_output_gradient + token_idx * headsAmount * headOutputSize + head * headOutputSize;
    const kernel_float* __restrict__ sm_cache   = cache_softmax         + token_idx * headsAmount * totalSamplesAmount + head * totalSamplesAmount + batch_start;

    // Pass 1: compute softmaxDot = sum_masked( softmax[masked] * dot(d_out, v[masked]) )
    kernel_float softmaxDot = 0.0f;
    for (int masked = 0; masked <= pos; ++masked) {
        int mt = batch_start + masked;
        const kernel_float* __restrict__ v = values + mt * headsAmount * headOutputSize + head * headOutputSize;

        kernel_float sm_grad = 0.0f;
        for (int e = 0; e < headOutputSize; ++e)
            sm_grad += d_out[e] * v[e];

        softmaxDot += sm_cache[masked] * sm_grad;
    }

    // Pass 2: accumulate queries_gradient using cached softmax + keys
    kernel_float q_grad = 0.0f;
    for (int masked = 0; masked <= pos; ++masked) {
        int mt = batch_start + masked;
        const kernel_float* __restrict__ k = keys   + mt * headsAmount * headOutputSize + head * headOutputSize;
        const kernel_float* __restrict__ v = values + mt * headsAmount * headOutputSize + head * headOutputSize;

        kernel_float sm_grad = 0.0f;
        for (int e = 0; e < headOutputSize; ++e)
            sm_grad += d_out[e] * v[e];

        kernel_float score_grad = sm_cache[masked] * (sm_grad - softmaxDot);
        q_grad += score_grad * k[head_dim] * scale;

        cache_score_gradients[token_idx * headsAmount * totalSamplesAmount + head * totalSamplesAmount + batch_start+masked] = score_grad;
    }

    queries_gradient[token_idx * headsAmount * headOutputSize + head * headOutputSize + head_dim] = q_grad;
}

extern "C" void multi_head_masked_self_attention_queries__gradient(const kernel_float* final_output_gradient
    , const kernel_float* keys, const kernel_float* values
    , const kernel_float* cache_softmax, kernel_float* cache_score_gradients
    , kernel_float* queries_gradient
    , const int& batchSize, const int* seriesLengths, const int& totalSamplesAmount
    , const int& headsAmount, const int& inputSize, const int& headOutputSize)
{
    dim3 threads(TILE, TILE, TILE);
    dim3 blocks(
        (headOutputSize + TILE - 1) / TILE,
        (headsAmount + TILE - 1) / TILE,
        (totalSamplesAmount + TILE - 1) / TILE
    );
    // launch kernel
    multi_head_masked_self_attention_queries__gradientKernel<<<blocks, threads>>>(
        final_output_gradient,
        keys, values,
        cache_softmax, cache_score_gradients,
        queries_gradient,
        batchSize, seriesLengths, totalSamplesAmount,
        headsAmount, inputSize, headOutputSize
    );
}