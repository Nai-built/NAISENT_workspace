#pragma once

#include "INeurologicalComponent.h"
#include "IActivationFunction.h"

struct TransformerStack {
    int attentionHeads;
    std::vector<int> hidden_FFN;
    std::vector<IActivationFunction> activations_FFN;
};

struct Transformer_DecodeOnly : public INeurologicalComponent
{
    int inputTokens;
    int outputTokens;

    int encodingSize;

    std::vector<TransformerStack> stacks;
    
    neurologicalConstantSpan cached_input;
    
    neurologicalBuffer encodeWeights;
    // positional encoding is considered its "biases", they're basically constant and not learnable

    neurologicalBuffer attentionWeights;
    neurologicalBuffer attentionFinalBiases;

    neurologicalBuffer decodeWeights;
    neurologicalBuffer decodeBiases;
    
    neurologicalBuffer encodeWeightsGradient;

    neurologicalBuffer attentionWeightsGradient;

    neurologicalBuffer decodeWeightsGradient;
    neurologicalBuffer decodeBiasesGradient;

    Transformer_DecodeOnly(int _inputTokens, int _outputTokens, int _encodingSize);

    static void Initialize(Transformer_DecodeOnly* transformer);
    static void ForwardPass(Transformer_DecodeOnly* transformer, const neurologicalConstantSpan input, const neurologicalSpan output, const int& batchSize
        , const lengths seriesLengths, const int& totalSamples);
    static void BackPropagation(Transformer_DecodeOnly* transformer, const neurologicalSpan propagation, const neurologicalSpan inputPropagation, const neurologicalValue& learnRate, const int& batchSize
        , const lengths seriesLengths, const int& totalSamples);

};