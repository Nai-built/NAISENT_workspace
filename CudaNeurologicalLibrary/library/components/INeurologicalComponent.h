#pragma once

#include "../cuda_Tensor.h"

enum class cuda_NeurologicalComponentTypes {
    COMPONENTS_CHAIN = 0,
    DENSE_LAYER = 1,
    RMSNORM_LAYER = 2,
    MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER = 3,
    SCC_POSITIONAL_EMBEDDING_LAYER = 4,
    CONVOLUTIONAL_LAYER = 5,
};

struct cuda_INeurologicalComponent {
    cuda_NeurologicalComponentTypes componentType;
    bool freeze   = false;
    bool residual = false;

    virtual ~cuda_INeurologicalComponent() = default; // Make the base polymorphic
};