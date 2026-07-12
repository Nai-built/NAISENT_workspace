#pragma once

#include "../optimization/IOptimization.h"

#include <memory>

#include "INeurologicalComponent.h"

struct cuda_SCCPositionalEmbeddingLayer : public cuda_INeurologicalComponent {
private:
    int inputSize;
    int outputSize;

public:
    // Parameters Optimization
    std::unique_ptr<cuda_IOptimization> weights_optimization;

    // Parameters
    cuda_Tensor weights;            // size = inputSize * outputSize  → layout: [in][out]

    // Gradients
    cuda_Tensor weights_gradient;   // size = inputSize * outputSize

    // Cache
    cuda_ConstantTensorSpan cached_input;

    // Alteration Embeddings
    cuda_Tensor sine_multipliers;
    cuda_Tensor cosine_multipliers;

    int inferred_amount;

    // Ctor / Dtor
    cuda_SCCPositionalEmbeddingLayer(int inSize, int outSize);

    // Size accessors
    int in_size() const;
    int out_size() const;

    // Logic
    static void Initialize(cuda_SCCPositionalEmbeddingLayer& layer, unsigned long seed);

    static void ForwardPass(
        cuda_SCCPositionalEmbeddingLayer& layer,
        cuda_ConstantTensorSpan input,
        cuda_TensorSpan output,
        const int& batchSize,
        lengths seriesLengths,
        const int& totalSamples
    );

    static void Infer(
        cuda_SCCPositionalEmbeddingLayer& layer,
        cuda_ConstantTensorSpan input,
        cuda_TensorSpan output,
        const int& sequenceLength
    );
    static void ClearInference(
        cuda_SCCPositionalEmbeddingLayer& layer
    );

    static void BackPropagation(
        cuda_SCCPositionalEmbeddingLayer& layer,
        cuda_ConstantTensorSpan output_gradient,
        cuda_TensorSpan input_gradient,
        const int& batchSize,
        lengths seriesLengths,
        const int& totalSamples
    );

    static void Update(cuda_SCCPositionalEmbeddingLayer& layer, const cuda_neurologicalValue& updateMultiplier);
};
