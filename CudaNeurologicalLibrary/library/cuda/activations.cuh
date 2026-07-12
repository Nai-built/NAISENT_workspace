#pragma once

#include "cuda_base.cuh"

extern "C" CUDA_API void conv_weights_linear__activation(const kernel_float* input, kernel_float* output
    , const kernel_float* weights
    , const int& samplesAmount, const int& inputChannels, const int& outputChannels

    , const int& inputWidth, const int& outputWidth
    , const int& inputHeight, const int& outputHeight

    , const int& slideWidth, const int& slideHeight

    , const int& stride, const int& padding);

extern "C" CUDA_API void fc_weights__activation(const kernel_float* input, kernel_float* output
    , const kernel_float* weights
    , const int& samplesAmount, const int& inputSize, const int& outputSize);

extern "C" CUDA_API void linear_af__activation(kernel_float* tensor
    , const kernel_float* biases
    , const int& samplesAmount, const int& tensorSize

    , const kernel_float& multiplier);
extern "C" CUDA_API void relu_af__activation(kernel_float* tensor
    , const kernel_float* biases
    , const int& samplesAmount, const int& tensorSize
    , kernel_float* cache_tensor, const kernel_float& fadeMultiplier

    , const kernel_float& multiplier);

extern "C" CUDA_API void norm_RMS__activation(const kernel_float* input_tensor
    , kernel_float* output_tensor, kernel_float* cache_rMultipliers
    , const kernel_float* gamma
    , const int& samplesAmount, const int& tensorSize);

extern "C" CUDA_API void multi_head_KVQ_weights__activation(const kernel_float* input
    , kernel_float* keys, kernel_float* values, kernel_float* qeuries
    , const kernel_float* weights
    , const int& samplesAmount, const int& headsAmount, const int& inputSize, const int& headOutputSize);

extern "C" CUDA_API void multi_head_masked_self_attention__activation(const kernel_float* keys, const kernel_float* values, const kernel_float* queries
    , kernel_float* final_output
    , kernel_float* cache_scores, kernel_float* cache_softmax
    , const int& batchSize, const int* seriesLengths
    , const int& headsAmount, const int& inputSize, const int& headOutputSize);

extern "C" CUDA_API void SCC_positional_embedding__activation(kernel_float* output
    , const kernel_float* sine_multipliers
    , const kernel_float* cosine_multipliers

    , const int& batchSize
    , const int* seriesLengths
    , const int& embeddingSize);

// INFERENCE!
extern "C" CUDA_API void SCC_positional_embedding_inference__activation(kernel_float* output
    , const kernel_float* sine_multipliers
    , const kernel_float* cosine_multipliers

    , const int& pre_length
    , const int& length
    , const int& embeddingSize);

extern "C" CUDA_API void MH_masked_self_attention_inference__activation(
    const kernel_float* keys,
    const kernel_float* values,
    const kernel_float* queries,
    kernel_float* final_output,
    kernel_float* cache_scores,
    kernel_float* cache_score_max,
    kernel_float* cache_exponent_sums,
    kernel_float* cache_softmax,
    const int& pre_length,
    const int& length,
    const int& headsAmount,
    const int& inputSize,
    const int& headOutputSize);