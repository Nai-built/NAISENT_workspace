#pragma once

#include "../activation_functions/IActivationFunction.h"
#include "../optimization/IOptimization.h"

#include <memory>
#include <random>

#include "INeurologicalComponent.h"

struct cuda_ConvolutionalLayer : public cuda_INeurologicalComponent {
private:
    int outputChannels;
    int inputChannels;
    int slideWidth;
    int slideHeight;
    int inputWidth;
    int inputHeight;
    int stride;
    int padding;
    int maxDropout;

public:

    // Activation Function
    std::unique_ptr<cuda_IActivationFunction> activation;

    // Parameters Optimization
    std::unique_ptr<cuda_IOptimization> weights_optimization;
    std::unique_ptr<cuda_IOptimization> biases_optimization;

    // Parameters
    cuda_Tensor weights;   // size = outputChannels * inputChannels * slideWidth * slideHeight
    cuda_Tensor biases;    // size = outputChannels

    // Gradients
    cuda_Tensor weights_gradient; // size = outputChannels * inputChannels * slideWidth * slideHeight
    cuda_Tensor biases_gradient;  // size = outputChannels

    // Cache
    cuda_ConstantTensorSpan cached_input;  // a span to the previously passed input

    // Extra
    cuda_Tensor dropout_gradient;
    cuda_SharedIntTensor dropout_mask;
    int droppedOut;
    std::mt19937 dropout_rand;
    std::uniform_int_distribution<int> dropout_rand_distribution;

    // Ctor / Dtor
    cuda_ConvolutionalLayer(
        int outChannels,
        int inChannels,
        int slideW,
        int slideH,
        int inW,
        int inH,
        int _stride,
        int _padding,
        int _maxDropout
    );

    // Size accessors
    int in_size() const;
    int out_size() const;

    int output_width() const;
    int output_height() const;

    // Logic
    // Static operations
    static void Initialize(cuda_ConvolutionalLayer& layer, unsigned long seed);

    static void ForwardPass(
        cuda_ConvolutionalLayer& layer,
        cuda_ConstantTensorSpan input,
        cuda_TensorSpan output,
        const int& samplesAmount
    );

    static void Infer(
        cuda_ConvolutionalLayer& layer,
        cuda_ConstantTensorSpan input,
        cuda_TensorSpan output,
        const int& sequenceLength
    );
    static void ClearInference(
        cuda_ConvolutionalLayer& layer
    );

    static void BackPropagation(
        cuda_ConvolutionalLayer& layer,
        cuda_ConstantTensorSpan output_gradient,
        cuda_TensorSpan input_gradient,
        const int& samplesAmount
    );

    static void Update(cuda_ConvolutionalLayer& layer, const cuda_neurologicalValue& updateMultiplier);
};
