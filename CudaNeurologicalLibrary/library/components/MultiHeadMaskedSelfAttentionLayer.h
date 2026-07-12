#pragma once

#include "../optimization/IOptimization.h"

#include <memory>

#include "INeurologicalComponent.h"

struct cuda_MultiHeadMaskedSelfAttentionLayer : public cuda_INeurologicalComponent {
private:
    int inputSize;
    int headOutputSize;
    int headsAmount;

public:
    // Parameters Optimization
    std::unique_ptr<cuda_IOptimization> weights_optimization;

    // Parameters
    cuda_Tensor weights;          // size = inputSize * headsAmount * headOutputSize * 3  (K, V, Q)

    // Gradients
    cuda_Tensor weights_gradient; // same size

    // Cache (forward)
    cuda_ConstantTensorSpan cached_input;
    cuda_Tensor cached_keys;          // totalSamples * headsAmount * headOutputSize
    cuda_Tensor cached_values;        // totalSamples * headsAmount * headOutputSize
    cuda_Tensor cached_queries;       // totalSamples * headsAmount * headOutputSize

    cuda_Tensor cache_scores;         // headsAmount * sum(seriesLengths[i]^2)
    cuda_Tensor cache_softmax;        // headsAmount * sum(seriesLengths[i]^2)
    
    cuda_Tensor cache_score_max;         // headsAmount * sum(seriesLengths[i]^2)
    cuda_Tensor cache_exponent_sums;        // headsAmount * sum(seriesLengths[i]^2)

    // Cache (backward)
    cuda_Tensor cached_keys_gradient;    // totalSamples * headsAmount * headOutputSize
    cuda_Tensor cached_values_gradient;  // totalSamples * headsAmount * headOutputSize
    cuda_Tensor cached_queries_gradient; // totalSamples * headsAmount * headOutputSize
    cuda_Tensor cache_scores_gradient;   // headsAmount * sum(seriesLengths[i]^2)

    int inferred_amount;

    // Ctor
    cuda_MultiHeadMaskedSelfAttentionLayer(int inputSize, int headOutputSize, int headsAmount);

    // Size accessors
    int in_size() const;
    int out_size() const;

    // Static operations
    static void Initialize(cuda_MultiHeadMaskedSelfAttentionLayer& layer, unsigned long seed);

    static void ForwardPass(
        cuda_MultiHeadMaskedSelfAttentionLayer& layer,
        cuda_ConstantTensorSpan input,
        cuda_TensorSpan output,
        const int& batchSize,
        lengths seriesLengths,
        const int& totalSamples
    );

    static void Infer(
        cuda_MultiHeadMaskedSelfAttentionLayer& layer,
        cuda_ConstantTensorSpan input,
        cuda_TensorSpan output,
        const int& sequenceLength
    );
    static void ClearInference(
        cuda_MultiHeadMaskedSelfAttentionLayer& layer
    );

    static void BackPropagation(
        cuda_MultiHeadMaskedSelfAttentionLayer& layer,
        cuda_ConstantTensorSpan output_gradient,
        cuda_TensorSpan input_gradient,
        const int& batchSize,
        lengths seriesLengths,
        const int& totalSamples
    );

    static void Update(cuda_MultiHeadMaskedSelfAttentionLayer& layer, const cuda_neurologicalValue& updateMultiplier);
};
