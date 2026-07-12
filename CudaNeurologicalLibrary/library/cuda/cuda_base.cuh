#pragma once

#ifdef _WIN32
  #ifdef CUDA_LIB_EXPORTS
    #define CUDA_API __declspec(dllexport)
  #else
    #define CUDA_API __declspec(dllimport)
  #endif
#else
  #define CUDA_API
#endif

#define TILE 8

typedef float kernel_float;

extern "C" CUDA_API void dropout(kernel_float* tensor
    , const int* dropout_mask
    , const int& tensorSize
    , const int& dropoutSize, const int& samplesAmount);

extern "C" CUDA_API void addition(kernel_float* main, const kernel_float* other
    , const int& mainSize, const int& otherSize);