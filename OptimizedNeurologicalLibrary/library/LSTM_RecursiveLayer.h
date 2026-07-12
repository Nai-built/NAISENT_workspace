#pragma once

#include "INeurologicalComponent.h"
#include "IActivationFunction.h"

struct LSTM_RecursiveLayer : public INeurologicalComponent
{
public:
    int inputSize;
    int outputSize;

    bool castBeyond;

    neurologicalConstantSpan cached_input;
    
    neurologicalBuffer weights;
    neurologicalBuffer biases;
    
    neurologicalBuffer weightsGradient;
    neurologicalBuffer biasesGradient;

    LSTM_RecursiveLayer(int _inputSize, int _outputSize, bool _castBeyond);

    static void Initialize(LSTM_RecursiveLayer* lstm);
    static void ForwardPass(LSTM_RecursiveLayer* lstm, const neurologicalConstantSpan input, const neurologicalSpan output, const int& batchSize
        , const lengths seriesLengths, const int& totalSamples);
    static void BackPropagation(LSTM_RecursiveLayer* lstm, const neurologicalSpan propagation, const neurologicalSpan inputPropagation, const neurologicalValue& learnRate, const int& batchSize
        , const lengths seriesLengths, const int& totalSamples);

    neurologicalBuffer hiddenStatePropagationBuffer;
    neurologicalBuffer memoryCellPropagationBuffer;
    
    size_t maxSamplesOutputSize;
    
    neurologicalBuffer hidden_state_buffer;
    neurologicalBuffer memory_cell_buffer;

    neurologicalBuffer forgetGate_buffer;
    neurologicalBuffer inputGate_buffer;
    neurologicalBuffer memoryGate_buffer;
    neurologicalBuffer outputGate_buffer;

    neurologicalBuffer memoryCell_activation;
};