#pragma once

#include "IActivationFunction.h"

struct cuda_ReLU : public cuda_IActivationFunction {
    cuda_neurologicalValue fadeMultiplier;

    cuda_Tensor cache_output;
    cuda_Tensor activation_gradient;
};