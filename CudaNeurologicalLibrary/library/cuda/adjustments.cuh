#pragma once

#include "cuda_base.cuh"

extern "C" CUDA_API void fc_weights__gradient(const kernel_float* output_gradient
    , const kernel_float* cached_input, kernel_float* weights_gradient
    , const int& samplesAmount, const int& inputSize, const int& outputSize);
extern "C" CUDA_API void fc_input__gradient(const kernel_float* output_gradient
    , const kernel_float* weights, kernel_float* input_gradient
    , const int& samplesAmount, const int& inputSize, const int& outputSize);

extern "C" CUDA_API void linear_af__gradient(const kernel_float* tensor_gradient
    , kernel_float* biases_gradient
    , const int& samplesAmount, const int& tensorSize);
extern "C" CUDA_API void relu_af__gradient(const kernel_float* tensor_gradient, kernel_float* activation_tensor_gradient
    , kernel_float* biases_gradient
    , const int& samplesAmount, const int& tensorSize
    , const kernel_float* cache_tensor, const kernel_float& fadeMultiplier);

extern "C" CUDA_API void gamma_RMS__gradient(const kernel_float* output_gradient
    , const kernel_float* cached_input, const kernel_float* cached_rMultipliers
    , kernel_float* gamma_gradient
    , const int& samplesAmount, const int& tensorSize);
extern "C" CUDA_API void input_RMS__gradient(const kernel_float* output_gradient
    , const kernel_float* cached_input, const kernel_float* cached_rMultipliers
    , const kernel_float* gamma
    , kernel_float* input_gradient
    , const int& samplesAmount, const int& tensorSize);

extern "C" CUDA_API void multi_head_KVQ_weights__gradient(const kernel_float* keys_gradient, const kernel_float* values_gradient, const kernel_float* qeuries_gradient
    , const kernel_float* cached_input, kernel_float* weights_gradient
    , const int& samplesAmount, const int& headsAmount, const int& inputSize, const int& headOutputSize);
extern "C" CUDA_API void multi_head_KVQ_input__gradient(const kernel_float* keys_gradient, const kernel_float* values_gradient, const kernel_float* qeuries_gradient
    , const kernel_float* weights, kernel_float* input_gradient
    , const int& samplesAmount, const int& headsAmount, const int& inputSize, const int& headOutputSize);

extern "C" CUDA_API void multi_head_masked_self_attention_KV__gradient(const kernel_float* final_output_gradient
    , const kernel_float* queries
    , const kernel_float* cache_softmax
    , const kernel_float* cache_scores_gradient
    , kernel_float* keys_gradient
    , kernel_float* values_gradient
    , const int& batchSize, const int* seriesLengths
    , const int& headsAmount, const int& inputSize, const int& headOutputSize);
extern "C" CUDA_API void multi_head_masked_self_attention_queries__gradient(const kernel_float* final_output_gradient
    , const kernel_float* keys
    , const kernel_float* values
    , const kernel_float* cache_softmax
    , kernel_float* cache_scores_gradient
    , kernel_float* queries_gradient
    , const int& batchSize, const int* seriesLengths
    , const int& headsAmount, const int& inputSize, const int& headOutputSize);