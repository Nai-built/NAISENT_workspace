#pragma once

#include "../INeurologicalComponent.h"
#include "../IActivationFunction.h"

struct ReLU : public IActivationFunction
{
public:
    neurologicalBuffer cached_output;

    neurologicalValue leakage;
    
    ReLU(neurologicalValue _leakage);
};