#pragma once

#include "cuda_base.cuh"

// extern "C" CUDA_API void default_dense__apply(const int& inputSize, const int& outputSize
//     , kernel_float* weights, kernel_float* biases
//     , const kernel_float* weights_gradient, const kernel_float* biases_gradient

//     , const kernel_float& multiplier);

extern "C" CUDA_API void default__apply(const int& size
    , kernel_float* parameters
    , const kernel_float* derivatives
    , const kernel_float& multiplier);

extern "C" CUDA_API void ADAM_optimization__apply(const int& size
    , const kernel_float& beta1, const kernel_float& beta2, const kernel_float& epsilon
    , int& t
    , kernel_float* m
    , kernel_float* v

    , kernel_float* parameters
    , const kernel_float* derivatives

    , const kernel_float& multiplier);