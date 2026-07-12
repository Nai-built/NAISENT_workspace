#pragma once

#include "INeurologicalComponent.h"
#include "IActivationFunction.h"

struct DecoderOnly_Transformer : public INeurologicalComponent
{
    int inputSize;
    int outputSize;
    int encodingSize; // hidden encoding values
    
    neurologicalConstantSpan cached_input;
    
    neurologicalBuffer weights;
    neurologicalBuffer biases;
    
    neurologicalBuffer weightsGradient;
    neurologicalBuffer biasesGradient;

    DecoderOnly_Transformer(int _inputSize, int _outputSize, int _encodingSize);

    static void Initialize(DecoderOnly_Transformer* transformer);
    static void ForwardPass(DecoderOnly_Transformer* transformer, const neurologicalConstantSpan input, const neurologicalSpan output, const int& batchSize
        , const lengths seriesLengths, const int& totalSamples);
    static void BackPropagation(DecoderOnly_Transformer* transformer, const neurologicalSpan propagation, const neurologicalSpan inputPropagation, const neurologicalValue& learnRate, const int& batchSize
        , const lengths seriesLengths, const int& totalSamples);

};