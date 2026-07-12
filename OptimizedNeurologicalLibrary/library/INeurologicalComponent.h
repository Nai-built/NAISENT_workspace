#pragma once

#include <vector>
#include <span>
#include <memory>

#include "Extra.h"

typedef float neurologicalValue;

typedef std::vector<neurologicalValue> neurologicalBuffer;
typedef std::span<neurologicalValue> neurologicalSpan;
typedef std::span<const neurologicalValue> neurologicalConstantSpan;

typedef std::span<const int> lengths;

enum class NeurologicalComponentTypes {
    COMPONENTS_CHAIN = 0,
    DENSE_LAYER = 1,
    CONVOLUTIONAL_LAYER = 2,
    RECURSIVE_LSTM = 3,
    DECODER_ONLY_TRANSFORMER = 4,
};

struct INeurologicalComponent
{
public:
    NeurologicalComponentTypes componentType;
    
    virtual ~INeurologicalComponent() = default; // Make the base polymorphic
};

typedef std::unique_ptr<INeurologicalComponent> NeurologicalComponent_UNIQUE;
typedef std::weak_ptr<INeurologicalComponent> NeurologicalComponentPTR;