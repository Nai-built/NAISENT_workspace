#pragma once

#include "INeurologicalComponent.h"
#include "IActivationFunction.h"

struct TransformerStack {
    int attentionHeads;
    std::vector<int> hidden_FFN;
    std::vector<ActivationFunction_UNIQUE> activations_FFN;
};

struct DecoderOnly_Transformer : public INeurologicalComponent
{
    int inputTokens;
    int outputTokens;

    int encodingSize;

    std::vector<TransformerStack> stacks;
    
    neurologicalConstantSpan cached_input;
    
    // cache buffers
    neurologicalBuffer ffns_buffer;

    neurologicalBuffer stream_encoding;
    
    neurologicalBuffer stream_encoding_stacks;
    neurologicalBuffer stream_normalization_stacks;
    neurologicalBuffer stream_attention_encoding_stacks;
    
    neurologicalBuffer keys_buffer;
    neurologicalBuffer queries_buffer;
    neurologicalBuffer values_buffer;

    neurologicalBuffer attention_scores;
    neurologicalBuffer attention_softmax;

    // propagation buffers
    neurologicalBuffer ffns_buffer_propagation;

    neurologicalBuffer stream_encoding_propagation;
    
    neurologicalBuffer stream_encoding_stacks_propagation;
    neurologicalBuffer stream_normalization_stacks_propagation;
    neurologicalBuffer stream_attention_encoding_stacks_propagation;
    
    neurologicalBuffer keys_buffer_propagation;
    neurologicalBuffer queries_buffer_propagation;
    neurologicalBuffer values_buffer_propagation;

    neurologicalBuffer attention_scores_tempGradient;
    neurologicalBuffer attention_softmax_tempGradient;

    // PARAMETERS BUFFERS
    neurologicalBuffer encodeWeights;
    // positional encoding is considered its "biases", they're basically constant and not learnable

    neurologicalBuffer attentionWeights;
    neurologicalBuffer attentionFinalBiases;

    neurologicalBuffer normalizationGammas;

    neurologicalBuffer feedForwardNetworksWeights;
    neurologicalBuffer feedForwardNetworksBiases;

    neurologicalBuffer decodeWeights;
    neurologicalBuffer decodeBiases;
    
    // GRADIENTS BUFFERS
    neurologicalBuffer encodeWeightsGradient;

    neurologicalBuffer attentionWeightsGradient;
    neurologicalBuffer attentionFinalBiasesGradient;

    neurologicalBuffer normalizationGammasGradient;

    neurologicalBuffer feedForwardNetworksWeightsGradient;
    neurologicalBuffer feedForwardNetworksBiasesGradient;

    neurologicalBuffer decodeWeightsGradient;
    neurologicalBuffer decodeBiasesGradient;

    DecoderOnly_Transformer(int _inputTokens, int _outputTokens, int _encodingSize);

    std::vector<double> positionalSineMultipliers;
    std::vector<double> positionalCosineMultipliers;

    int sineAlternationLength;
    int cosineAlternationLength;

    int ffns_buffer_size;

    static void Initialize(DecoderOnly_Transformer* transformer);
    static void ForwardPass(DecoderOnly_Transformer* transformer, const neurologicalConstantSpan input, const neurologicalSpan output, const int& batchSize
        , const lengths seriesLengths, const int& totalSamples);
    static void BackPropagation(DecoderOnly_Transformer* transformer, const neurologicalSpan propagation, const neurologicalSpan inputPropagation, const neurologicalValue& learnRate, const int& batchSize
        , const lengths seriesLengths, const int& totalSamples);
};