#pragma once

#include "../cuda_Tensor.h"

enum class cuda_OptimizerTypes {
    DEFAULT = 0,
    ADAM = 1,
};

struct cuda_IOptimization {
    cuda_OptimizerTypes optimizerType;
    
    virtual ~cuda_IOptimization() = default; // Make the base polymorphic
};