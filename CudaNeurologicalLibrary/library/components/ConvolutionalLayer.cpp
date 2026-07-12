#include "ConvolutionalLayer.h"

#include "../cuda/activations.cuh"
#include "../cuda/adjustments.cuh"
#include "../cuda/initialization.cuh"
#include "../cuda/optimization.cuh"

#include "../activation_functions/ReLU.h"
#include "../optimization/ADAM_optimization.h"

#include <cuda_runtime.h>

// Constructor
cuda_ConvolutionalLayer::cuda_ConvolutionalLayer(
    int outChannels,
    int inChannels,
    int slideW,
    int slideH,
    int inW,
    int inH,
    int _stride,
    int _padding,
    int _maxDropout
)
    : outputChannels(outChannels),
      inputChannels(inChannels),
      slideWidth(slideW),
      slideHeight(slideH),
      inputWidth(inW),
      inputHeight(inH),
      stride(_stride),
      padding(_padding),
      weights(outChannels * inChannels * slideW * slideH),
      biases(outChannels),
      weights_gradient(outChannels * inChannels * slideW * slideH),
      biases_gradient(outChannels),

      maxDropout(_maxDropout),
      dropout_mask(_maxDropout)
{
    // 1. Obtain a random seed from hardware for drop out masking
    std::random_device rd;

    // 2. Initialize the Mersenne Twister engine with the seed for drop out masking
    std::mt19937 gen(rd());

    // 3. Define the range [min, max] (both inclusive)
    std::uniform_int_distribution<int> distrib(0, this->out_size()-1);

    this->dropout_rand = gen;
    this->dropout_rand_distribution = distrib;
}

// Size accessors
int cuda_ConvolutionalLayer::in_size() const {
    return inputChannels * inputWidth * inputHeight;
}

int cuda_ConvolutionalLayer::out_size() const {
    return outputChannels * output_width() * output_height();
}

int cuda_ConvolutionalLayer::output_width() const {
    return (inputWidth + 2*padding - slideWidth) / stride + 1;
}

int cuda_ConvolutionalLayer::output_height() const {
    return (inputHeight + 2*padding - slideHeight) / stride + 1;
}

// Static: Initialize
void cuda_ConvolutionalLayer::Initialize(cuda_ConvolutionalLayer& layer, unsigned long seed) {
    // TODO: kernel launcher for convolutional weights initialization
    conv_weights__initialization(
        layer.inputChannels, layer.outputChannels,
        layer.slideWidth, layer.slideHeight,
        layer.weights.data(),
        seed
    );
}

// Static: Forward Pass
void cuda_ConvolutionalLayer::ForwardPass(
    cuda_ConvolutionalLayer& layer,
    cuda_ConstantTensorSpan input,
    cuda_TensorSpan output,
    const int& samplesAmount
) {
    layer.cached_input = input;

    conv_weights_linear__activation(
        input.data(),
        output.data(),
        layer.weights.data(),
        samplesAmount,
        layer.inputChannels, layer.outputChannels,

        layer.inputWidth, layer.output_width(),
        layer.inputHeight, layer.output_height(),

        layer.slideWidth, layer.slideHeight,
        
        layer.stride, layer.padding
    );
    cudaDeviceSynchronize();

    // TODO: kernel launcher for convolutional forward pass (activation + bias + activation function)
    // TODO: dropout handling once a kernel launcher is wired in

    layer.droppedOut = 0;
}

// Static: Infer
void cuda_ConvolutionalLayer::Infer(
    cuda_ConvolutionalLayer& layer,
    cuda_ConstantTensorSpan input,
    cuda_TensorSpan output,
    const int& sequenceLength
) {
    // TODO: kernel launcher for convolutional inference (activation + bias + activation function)
}
void cuda_ConvolutionalLayer::ClearInference(cuda_ConvolutionalLayer& layer) {
    // EMPTY
}

// Static: Backpropagation
void cuda_ConvolutionalLayer::BackPropagation(
    cuda_ConvolutionalLayer& layer,
    cuda_ConstantTensorSpan output_gradient,
    cuda_TensorSpan input_gradient,
    const int& samplesAmount
) {
    // TODO: kernel launcher for convolutional back propagation
    //  - propagate through activation function gradient
    //  - compute weights_gradient and biases_gradient
    //  - compute input_gradient
}

// Static: Apply parameter update
void cuda_ConvolutionalLayer::Update(cuda_ConvolutionalLayer& layer, const cuda_neurologicalValue& updateMultiplier) {
    if (layer.biases_optimization != nullptr) {
        switch(layer.biases_optimization->optimizerType) {
            case cuda_OptimizerTypes::ADAM: {
                cuda_ADAM_optimization* adam = static_cast<cuda_ADAM_optimization*>(layer.biases_optimization.get());
                ADAM_optimization__apply(
                    layer.biases.size,
                    adam->beta1,
                    adam->beta2,
                    adam->epsilon,
                    adam->t,
                    adam->m.data(),
                    adam->v.data(),
                    layer.biases.data(),
                    layer.biases_gradient.data(),
                    updateMultiplier
                );
                break;
            }
            default: {
                default__apply(
                    layer.biases.size,
                    layer.biases.data(),
                    layer.biases_gradient.data(),
                    updateMultiplier
                );
                break;
            }
        }
    } else {
        default__apply(
            layer.biases.size,
            layer.biases.data(),
            layer.biases_gradient.data(),
            updateMultiplier
        );
    }
    if (layer.weights_optimization != nullptr) {
        switch(layer.weights_optimization->optimizerType) {
            case cuda_OptimizerTypes::ADAM: {
                cuda_ADAM_optimization* adam = static_cast<cuda_ADAM_optimization*>(layer.weights_optimization.get());
                ADAM_optimization__apply(
                    layer.weights.size,
                    adam->beta1,
                    adam->beta2,
                    adam->epsilon,
                    adam->t,
                    adam->m.data(),
                    adam->v.data(),
                    layer.weights.data(),
                    layer.weights_gradient.data(),
                    updateMultiplier
                );
                break;
            }
            default: {
                default__apply(
                    layer.weights.size,
                    layer.weights.data(),
                    layer.weights_gradient.data(),
                    updateMultiplier
                );
                break;
            }
        }
    } else {
        default__apply(
            layer.weights.size,
            layer.weights.data(),
            layer.weights_gradient.data(),
            updateMultiplier
        );
    }
    cudaDeviceSynchronize();
    layer.weights_gradient.zeroAll();
    layer.biases_gradient.zeroAll();
}
