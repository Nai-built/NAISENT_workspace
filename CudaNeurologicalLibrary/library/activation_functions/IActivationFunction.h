#pragma once

#include "../cuda_Tensor.h"

enum class cuda_ActivationFunctionTypes {
    LINEAR = 0,
    ReLU = 1,
};

struct cuda_IActivationFunction {
    cuda_ActivationFunctionTypes activationType;
    
    virtual ~cuda_IActivationFunction() = default; // Make the base polymorphic
};