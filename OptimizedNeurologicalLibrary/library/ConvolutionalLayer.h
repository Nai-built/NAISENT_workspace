#pragma once

#include "INeurologicalComponent.h"
#include "IActivationFunction.h"
#include "IChannelsPooler.h"

struct ConvolutionalLayer : public INeurologicalComponent
{
    int inputChannels;
    int outputChannels;

    int padding;
    int stride;

    int kernelWidth;
    int kernelHeight;

    int inputChannelWidth;
    int inputChannelHeight;

    int outputChannelWidth;
    int outputChannelHeight;

    int finalChannelWidth;
    int finalChannelHeight;

    int inputSize;
    int outputSize;
    int finalOutputSize;

    neurologicalConstantSpan cached_input;
    
    neurologicalBuffer weights;
    neurologicalBuffer biases;
    
    neurologicalBuffer weightsGradient;
    neurologicalBuffer biasesGradient;

    ActivationFunction_UNIQUE activationFunction;
    ChannelsPooler_UNIQUE channelsPooler;

    ConvolutionalLayer(int _inputChannels, int _outputChannels
        , int _kernelWidth, int _kernelHeight
        , int _inputChannelWidth, int _inputChannelHeight
        , int _stride, int _padding
        , ActivationFunction_UNIQUE _activationFunction
        , ChannelsPooler_UNIQUE _channelsPooler);

    static void Initialize(ConvolutionalLayer* convolutionalLayer);
    static void ForwardPass(ConvolutionalLayer* convolutionalLayer, const neurologicalConstantSpan input, const neurologicalSpan output, const int& batchSize);
    static void BackPropagation(ConvolutionalLayer* convolutionalLayer, const neurologicalSpan propagation, const neurologicalSpan inputPropagation, const neurologicalValue& learnRate, const int& batchSize);

    
    neurologicalBuffer packedPropagation;
    size_t packedCap = 0;
};