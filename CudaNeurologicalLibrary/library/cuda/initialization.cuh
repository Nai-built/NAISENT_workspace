#pragma once

#include "cuda_base.cuh"

extern "C" CUDA_API void fc_weights__initialization(const int& inputSize, const int& outputSize
    , kernel_float* weights

    , unsigned long initializationSeed);
extern "C" CUDA_API void conv_weights__initialization(const int& inputChannels, const int& outputChannels
    , const int& slideWidth, const int& slideHeight
    , kernel_float* weights

    , unsigned long initializationSeed);
extern "C" CUDA_API void rms__initialization(const int& tensorSize
    , kernel_float* gamma

    , unsigned long initializationSeed);
extern "C" CUDA_API void multi_head_KVQ__initialization(const int& inputSize, const int& headsAmount, const int& headOutputSize
    , kernel_float* weights

    , unsigned long initializationSeed);

extern "C" CUDA_API void sine_cosine_cycle__initialization(const int& embeddingSize
    , kernel_float* sine_multipliers
    , kernel_float* cosine_multipliers);