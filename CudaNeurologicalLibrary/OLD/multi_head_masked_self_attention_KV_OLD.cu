#include "../../adjustments.cuh"

#include <iostream>

__global__ void multi_head_masked_self_attention_KV__gradientKernel(
    const kernel_float* __restrict__ final_output_gradient,
    const kernel_float* __restrict__ queries,
    const kernel_float* __restrict__ cache_score_gradients,
    const kernel_float* __restrict__ cache_softmax,
    kernel_float* __restrict__ keys_gradient,
    kernel_float* __restrict__ values_gradient,
    const int batchSize, const int* __restrict__ seriesLengths, const int totalSamplesAmount,
    const int headsAmount, const int inputSize, const int headOutputSize)
{
    int head_dim     = blockIdx.x * TILE + threadIdx.x;
    int head         = blockIdx.y * TILE + threadIdx.y;
    int masked_token = blockIdx.z * TILE + threadIdx.z;  // z = masked, not pos

    if (head_dim >= headOutputSize || head >= headsAmount || masked_token >= totalSamplesAmount) return;

    // find batch and local masked position
    int batch_start = 0;
    int length = 0;
    for (int b = 0; b < batchSize; ++b) {
        int l = seriesLengths[b];
        if (batch_start + l > masked_token) { length = l; break; }
        batch_start += l;
    }
    int masked = masked_token - batch_start;

    const kernel_float scale = rsqrtf((kernel_float)headOutputSize);

    kernel_float k_grad = 0.0f;
    kernel_float v_grad = 0.0f;

    for (int pos_local = masked; pos_local < length; ++pos_local) {
        int pos_global = batch_start + pos_local;

        const int ptr_indexing = pos_global * headsAmount * headOutputSize + head * headOutputSize;
        const kernel_float* __restrict__ d_out = final_output_gradient + ptr_indexing;
        const kernel_float* __restrict__ q     = queries               + ptr_indexing;

        // dense cache layout: [pos][head][totalSamplesAmount], offset to batch_start for masked index
        const int grad_indexing = pos_global * headsAmount * totalSamplesAmount + head * totalSamplesAmount + masked_token;
        const kernel_float sm_weight   = cache_softmax        [grad_indexing];
        const kernel_float score_grad  = cache_score_gradients[grad_indexing];

        v_grad += sm_weight  * d_out[head_dim];
        k_grad += score_grad * q[head_dim] * scale;
    }

    const int indexing = masked_token * headsAmount * headOutputSize + head * headOutputSize + head_dim;
    keys_gradient  [indexing] = k_grad;
    values_gradient[indexing] = v_grad;
}

extern "C" void multi_head_masked_self_attention_KV__gradient(const kernel_float* final_output_gradient
    , const kernel_float* queries
    , const kernel_float* cache_score_gradients, const kernel_float* cache_softmax
    , kernel_float* keys_gradient, kernel_float* values_gradient
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
    multi_head_masked_self_attention_KV__gradientKernel<<<blocks, threads>>>(
        final_output_gradient,
        queries,
        cache_score_gradients, cache_softmax,
        keys_gradient, values_gradient,
        batchSize, seriesLengths, totalSamplesAmount,
        headsAmount, inputSize, headOutputSize
    );
}