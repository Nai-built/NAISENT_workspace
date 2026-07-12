#pragma once

#include <vector>
#include <span>
#include <memory>

enum class ActivationFunctionTypes {
    LINEAR = 0,

    RELU = 1,
};

struct IActivationFunction
{
public:
    ActivationFunctionTypes activationType;
    
    virtual ~IActivationFunction() = default; // Make the base polymorphic
};

typedef std::unique_ptr<IActivationFunction> ActivationFunction_UNIQUE;
typedef std::weak_ptr<IActivationFunction> ActivationFunctionPTR;