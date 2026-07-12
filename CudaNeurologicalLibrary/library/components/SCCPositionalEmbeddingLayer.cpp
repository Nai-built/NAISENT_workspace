#include "SCCPositionalEmbeddingLayer.h"

#include "../optimization/ADAM_optimization.h"
#include "../cuda/optimization.cuh"
#include "../cuda/initialization.cuh"
#include "../cuda/activations.cuh"
#include "../cuda/adjustments.cuh"

#include <cuda_runtime.h>

#include <iostream>

// Constructor
cuda_SCCPositionalEmbeddingLayer::cuda_SCCPositionalEmbeddingLayer(int inSize, int outSize)
    : inputSize(inSize),
      outputSize(outSize),
      weights(inSize * outSize),
      weights_gradient(inSize * outSize),

      sine_multipliers(std::ceil((float)(outSize/2))),
      cosine_multipliers(std::floor((float)(outSize/2))),

      inferred_amount(0)
{
}

// Size accessors
int cuda_SCCPositionalEmbeddingLayer::in_size() const {
    return inputSize;
}

int cuda_SCCPositionalEmbeddingLayer::out_size() const {
    return outputSize;
}

// Static: Initialize (kernel added later)
void cuda_SCCPositionalEmbeddingLayer::Initialize(cuda_SCCPositionalEmbeddingLayer& layer, unsigned long seed) {
    fc_weights__initialization(
        layer.inputSize,
        layer.outputSize,
        layer.weights.data(),
        seed
    );
    sine_cosine_cycle__initialization(
        layer.outputSize,
        layer.sine_multipliers.data(),
        layer.cosine_multipliers.data()
    );
}

// Static: Forward Pass (kernel added later)
void cuda_SCCPositionalEmbeddingLayer::ForwardPass(
    cuda_SCCPositionalEmbeddingLayer& layer,
    cuda_ConstantTensorSpan input,
    cuda_TensorSpan output,
    const int& batchSize,
    lengths seriesLengths,
    const int& totalSamples
) {
    layer.cached_input = input;
    fc_weights__activation(
        input.data(),
        output.data(),
        layer.weights.data(),
        totalSamples,
        layer.inputSize,
        layer.outputSize
    );
    cudaDeviceSynchronize();
    SCC_positional_embedding__activation(
        output.data(),
        layer.sine_multipliers.data(),
        layer.cosine_multipliers.data(),
        batchSize,
        seriesLengths.data(),
        layer.out_size()
    );
    cudaDeviceSynchronize();
}

// Static: Infer
void cuda_SCCPositionalEmbeddingLayer::Infer(
    cuda_SCCPositionalEmbeddingLayer& layer,
    cuda_ConstantTensorSpan input,
    cuda_TensorSpan output,
    const int& sequenceLength
) {
    fc_weights__activation(
        input.data(),
        output.data(),
        layer.weights.data(),
        sequenceLength,
        layer.inputSize,
        layer.outputSize
    );
    cudaDeviceSynchronize();
    SCC_positional_embedding_inference__activation(
        output.data(),
        layer.sine_multipliers.data(),
        layer.cosine_multipliers.data(),
        layer.inferred_amount,
        sequenceLength,
        layer.out_size()
    );
    cudaDeviceSynchronize();

    layer.inferred_amount += sequenceLength;
}
void cuda_SCCPositionalEmbeddingLayer::ClearInference(cuda_SCCPositionalEmbeddingLayer& layer) {
    layer.inferred_amount = 0;
}

// Static: Backpropagation (kernel added later)
void cuda_SCCPositionalEmbeddingLayer::BackPropagation(
    cuda_SCCPositionalEmbeddingLayer& layer,
    cuda_ConstantTensorSpan output_gradient,
    cuda_TensorSpan input_gradient,
    const int& batchSize,
    lengths seriesLengths,
    const int& totalSamples
) {
    fc_weights__gradient(
        output_gradient.data(),
        layer.cached_input.data(),
        layer.weights_gradient.data(),
        totalSamples,
        layer.inputSize,
        layer.outputSize
    );
    fc_input__gradient(
        output_gradient.data(),
        layer.weights.data(),
        input_gradient.data(),
        totalSamples,
        layer.inputSize,
        layer.outputSize
    );
    cudaDeviceSynchronize();
}

// Static: Apply parameter update
void cuda_SCCPositionalEmbeddingLayer::Update(cuda_SCCPositionalEmbeddingLayer& layer, const cuda_neurologicalValue& updateMultiplier) {
    if (layer.weights_optimization != nullptr) {
        switch (layer.weights_optimization->optimizerType) {
            case cuda_OptimizerTypes::ADAM: {
                cuda_ADAM_optimization* adam = static_cast<cuda_ADAM_optimization*>(layer.weights_optimization.get());
                ADAM_optimization__apply(
                    layer.in_size() * layer.out_size(),
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
                    layer.in_size() * layer.out_size(),
                    layer.weights.data(),
                    layer.weights_gradient.data(),
                    updateMultiplier
                );
                break;
            }
        }
    } else {
        default__apply(
            layer.in_size() * layer.out_size(),
            layer.weights.data(),
            layer.weights_gradient.data(),
            updateMultiplier
        );
    }
    cudaDeviceSynchronize();
    layer.weights_gradient.zeroAll();
}
