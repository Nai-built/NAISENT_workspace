#pragma once

#include "IOptimization.h"

struct cuda_ADAM_optimization : public cuda_IOptimization {
    int t;

    cuda_neurologicalValue beta1;
    cuda_neurologicalValue beta2;
    cuda_neurologicalValue epsilon;

    cuda_Tensor m;
    cuda_Tensor v;
};