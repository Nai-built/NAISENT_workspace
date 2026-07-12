#include "MultiHeadMaskedSelfAttentionLayer.h"

#include "../cuda/activations.cuh"
#include "../cuda/adjustments.cuh"
#include "../cuda/optimization.cuh"
#include "../cuda/initialization.cuh"

#include "../optimization/ADAM_optimization.h"

#include <cuda_runtime.h>

// Constructor
cuda_MultiHeadMaskedSelfAttentionLayer::cuda_MultiHeadMaskedSelfAttentionLayer(int inputSize, int headOutputSize, int headsAmount)
    : inputSize(inputSize),
      headOutputSize(headOutputSize),
      headsAmount(headsAmount),
      weights(inputSize * headsAmount * headOutputSize * 3),
      weights_gradient(inputSize * headsAmount * headOutputSize * 3),

      inferred_amount(0)
{
}

// Size accessors
int cuda_MultiHeadMaskedSelfAttentionLayer::in_size() const {
    return inputSize;
}

int cuda_MultiHeadMaskedSelfAttentionLayer::out_size() const {
    return headsAmount * headOutputSize;
}

// Static: Initialize (weights zeroed — no init kernel yet)
void cuda_MultiHeadMaskedSelfAttentionLayer::Initialize(cuda_MultiHeadMaskedSelfAttentionLayer& layer, unsigned long seed) {
    multi_head_KVQ__initialization(
        layer.inputSize,
        layer.headsAmount, layer.headOutputSize,
        layer.weights.data(),
        seed
    );
}

// Static: Forward Pass
void cuda_MultiHeadMaskedSelfAttentionLayer::ForwardPass(
    cuda_MultiHeadMaskedSelfAttentionLayer& layer,
    cuda_ConstantTensorSpan input,
    cuda_TensorSpan output,
    const int& batchSize,
    lengths seriesLengths,
    const int& totalSamples
) {
    layer.cached_input = input;

    // Lazily resize KVQ intermediate buffers
    const int kvqSize = totalSamples * layer.headsAmount * layer.headOutputSize;
    if (layer.cached_keys.size < kvqSize) {
        layer.cached_keys    = cuda_Tensor(kvqSize);
        layer.cached_values  = cuda_Tensor(kvqSize);
        layer.cached_queries = cuda_Tensor(kvqSize);
    }
    layer.cached_keys.zeroAll();
    layer.cached_values.zeroAll();
    layer.cached_queries.zeroAll();

    // Lazily resize score caches
    const int scoresSize = layer.headsAmount * totalSamples * totalSamples;
    if (layer.cache_scores.size < scoresSize) {
        layer.cache_scores  = cuda_Tensor(scoresSize);
        layer.cache_softmax = cuda_Tensor(scoresSize);
    }
    layer.cache_scores.zeroAll();
    layer.cache_softmax.zeroAll();
    const int score_maxSize = layer.headsAmount * totalSamples;
    if (layer.cache_score_max.size < score_maxSize) {
        layer.cache_score_max  = cuda_Tensor(score_maxSize);
        layer.cache_exponent_sums = cuda_Tensor(score_maxSize);
    }
    layer.cache_score_max.zeroAll();
    layer.cache_exponent_sums.zeroAll();

    // Project input → K, V, Q
    multi_head_KVQ_weights__activation(
        input.data(),
        layer.cached_keys.data(),
        layer.cached_values.data(),
        layer.cached_queries.data(),
        layer.weights.data(),
        totalSamples,
        layer.headsAmount,
        layer.inputSize,
        layer.headOutputSize
    );
    cudaDeviceSynchronize();

    // Masked self-attention
    multi_head_masked_self_attention__activation(
        layer.cached_keys.data(),
        layer.cached_values.data(),
        layer.cached_queries.data(),
        output.data(),
        layer.cache_scores.data(),
        layer.cache_softmax.data(),
        batchSize,
        seriesLengths.data(),
        layer.headsAmount,
        layer.inputSize,
        layer.headOutputSize
    );
    cudaDeviceSynchronize();
}

// Static: Infer
void cuda_MultiHeadMaskedSelfAttentionLayer::Infer(
    cuda_MultiHeadMaskedSelfAttentionLayer& layer,
    cuda_ConstantTensorSpan input,
    cuda_TensorSpan output,
    const int& sequenceLength
) {
    const int bufferSize = sequenceLength * layer.headsAmount * layer.headOutputSize;
    if (layer.cached_queries.size < bufferSize) {
        layer.cached_queries = cuda_Tensor(bufferSize);
    }
    layer.cached_queries.zeroAll();

    const int scoresSize = layer.headsAmount * sequenceLength * (layer.inferred_amount+sequenceLength);
    if (layer.cache_scores.size < scoresSize) {
        layer.cache_scores  = cuda_Tensor(scoresSize);
        layer.cache_softmax = cuda_Tensor(scoresSize);
    }
    layer.cache_scores.zeroAll();
    layer.cache_softmax.zeroAll();
    const int score_maxSize = layer.headsAmount * sequenceLength;
    if (layer.cache_score_max.size < score_maxSize) {
        layer.cache_score_max  = cuda_Tensor(score_maxSize);
        layer.cache_exponent_sums = cuda_Tensor(score_maxSize);
    }
    layer.cache_score_max.zeroAll();
    layer.cache_exponent_sums.zeroAll();

    const int kvPass = layer.inferred_amount * layer.headsAmount * layer.headOutputSize;

    if (layer.cached_keys.size < kvPass+bufferSize) {
        layer.cached_keys.resize(kvPass+bufferSize);
        layer.cached_values.resize(kvPass+bufferSize);
    }

    multi_head_KVQ_weights__activation(
        input.data(),
        layer.cached_keys.data()+kvPass,
        layer.cached_values.data()+kvPass,
        layer.cached_queries.data(),
        layer.weights.data(),
        sequenceLength,
        layer.headsAmount,
        layer.inputSize,
        layer.headOutputSize
    );
    cudaDeviceSynchronize();

    // multi_head_masked_self_attention__activation(
    //     layer.cached_keys.data(),
    //     layer.cached_values.data(),
    //     layer.cached_queries.data(),
    //     output.data(),
    //     layer.cache_scores.data(),
    //     layer.cache_softmax.data(),
    //     1,
    //     &sequenceLength,
    //     layer.headsAmount,
    //     layer.inputSize,
    //     layer.headOutputSize
    // );
    MH_masked_self_attention_inference__activation(
        layer.cached_keys.data(),
        layer.cached_values.data(),
        layer.cached_queries.data(),
        output.data(),
        layer.cache_scores.data(),
        layer.cache_score_max.data(),
        layer.cache_exponent_sums.data(),
        layer.cache_softmax.data(),
        layer.inferred_amount,
        sequenceLength,
        layer.headsAmount,
        layer.inputSize,
        layer.headOutputSize
    );
    cudaDeviceSynchronize();

    layer.inferred_amount += sequenceLength;
}
void cuda_MultiHeadMaskedSelfAttentionLayer::ClearInference(cuda_MultiHeadMaskedSelfAttentionLayer& layer) {
    layer.inferred_amount = 0;
    layer.cached_keys.zeroAll();
    layer.cached_values.zeroAll();
}

// Static: Backpropagation
void cuda_MultiHeadMaskedSelfAttentionLayer::BackPropagation(
    cuda_MultiHeadMaskedSelfAttentionLayer& layer,
    cuda_ConstantTensorSpan output_gradient,
    cuda_TensorSpan input_gradient,
    const int& batchSize,
    lengths seriesLengths,
    const int& totalSamples
) {
    // Lazily resize KVQ gradient buffers (reuse cached_keys/values/queries storage for gradients)
    const int kvqSize = totalSamples * layer.headsAmount * layer.headOutputSize;
    if (layer.cached_keys_gradient.size < kvqSize) {
        layer.cached_keys_gradient    = cuda_Tensor(kvqSize);
        layer.cached_values_gradient  = cuda_Tensor(kvqSize);
        layer.cached_queries_gradient = cuda_Tensor(kvqSize);
    }
    layer.cached_keys_gradient.zeroAll();
    layer.cached_values_gradient.zeroAll();
    layer.cached_queries_gradient.zeroAll();

    // printf("buffer: %i; gradient: %i", layer.cache);

    // Lazily resize scores gradient buffer
    const int scoresSize = layer.headsAmount * totalSamples * totalSamples;
    if (layer.cache_scores_gradient.size < scoresSize) {
        layer.cache_scores_gradient = cuda_Tensor(scoresSize);
    }
    layer.cache_scores_gradient.zeroAll();

    // Step 1: queries gradient (also fills cache_scores_gradient)
    multi_head_masked_self_attention_queries__gradient(
        output_gradient.data(),
        layer.cached_keys.data(),
        layer.cached_values.data(),
        layer.cache_softmax.data(),
        layer.cache_scores_gradient.data(),
        layer.cached_queries_gradient.data(),   // output: queries gradient
        batchSize,
        seriesLengths.data(),
        layer.headsAmount,
        layer.inputSize,
        layer.headOutputSize
    );
    cudaDeviceSynchronize();

    // Step 2: K/V gradients (reads cache_scores_gradient filled above)
    multi_head_masked_self_attention_KV__gradient(
        output_gradient.data(),
        layer.cached_queries.data(),   // queries gradient from step 1
        layer.cache_softmax.data(),
        layer.cache_scores_gradient.data(),
        layer.cached_keys_gradient.data(),      // output: keys gradient
        layer.cached_values_gradient.data(),    // output: values gradient
        batchSize,
        seriesLengths.data(),
        layer.headsAmount,
        layer.inputSize,
        layer.headOutputSize
    );
    cudaDeviceSynchronize();

    // Step 3: weights gradient (from K/V/Q gradients and cached input)
    multi_head_KVQ_weights__gradient(
        layer.cached_keys_gradient.data(),
        layer.cached_values_gradient.data(),
        layer.cached_queries_gradient.data(),
        layer.cached_input.data(),
        layer.weights_gradient.data(),
        totalSamples,
        layer.headsAmount,
        layer.inputSize,
        layer.headOutputSize
    );

    // Step 4: input gradient
    multi_head_KVQ_input__gradient(
        layer.cached_keys_gradient.data(),
        layer.cached_values_gradient.data(),
        layer.cached_queries_gradient.data(),
        layer.weights.data(),
        input_gradient.data(),
        totalSamples,
        layer.headsAmount,
        layer.inputSize,
        layer.headOutputSize
    );
    cudaDeviceSynchronize();
}

// Static: Apply parameter update
void cuda_MultiHeadMaskedSelfAttentionLayer::Update(cuda_MultiHeadMaskedSelfAttentionLayer& layer, const cuda_neurologicalValue& updateMultiplier) {
    if (layer.weights_optimization != nullptr) {
        switch (layer.weights_optimization->optimizerType) {
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
}
