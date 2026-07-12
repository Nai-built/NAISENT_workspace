#pragma once

#include "INeurologicalComponent.h"
#include "IActivationFunction.h"

struct DenseLayer : public INeurologicalComponent
{
public:
    int inputSize;
    int outputSize;

    neurologicalConstantSpan cached_input;
    
    neurologicalBuffer weights;
    neurologicalBuffer biases;
    
    neurologicalBuffer weightsGradient;
    neurologicalBuffer biasesGradient;

    ActivationFunction_UNIQUE activationFunction;

    DenseLayer(int _inputSize, int _outputSize, ActivationFunction_UNIQUE _activationFunction);

    static void Initialize(DenseLayer* denseLayer);
    static void ForwardPass(DenseLayer* denseLayer, const neurologicalConstantSpan input, const neurologicalSpan output, const int& batchSize);
    static void BackPropagation(DenseLayer* denseLayer, const neurologicalSpan propagation, const neurologicalSpan inputPropagation, const neurologicalValue& learnRate, const int& batchSize);
};