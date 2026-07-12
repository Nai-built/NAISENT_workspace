#include "BuildConfigurations.h"

// Base Optimizer
BaseOptimizer__BuildInfo::BaseOptimizer__BuildInfo() {
    optimizerType = cuda_OptimizerTypes::DEFAULT;
}

// ADAM
ADAM__BuildInfo::ADAM__BuildInfo() {
    optimizerType = cuda_OptimizerTypes::ADAM;
}

// Base Component
BaseNeurologicalComponent__BuildInfo::BaseNeurologicalComponent__BuildInfo()
    : freeze(false), residual(false) {}

// Base Activation
BaseActivationFunction__BuildInfo::BaseActivationFunction__BuildInfo() {}

// ReLU
ReLU__BuildInfo::ReLU__BuildInfo() {
    componentType = cuda_ActivationFunctionTypes::ReLU;
}

// RMSNorm
RMSNormLayer__BuildInfo::RMSNormLayer__BuildInfo() {
    componentType = cuda_NeurologicalComponentTypes::RMSNORM_LAYER; // make sure enum exists
}
// Dense
DenseLayer__BuildInfo::DenseLayer__BuildInfo() {
    componentType = cuda_NeurologicalComponentTypes::DENSE_LAYER;
}

// Convolutional
ConvolutionalLayer__BuildInfo::ConvolutionalLayer__BuildInfo() {
    componentType = cuda_NeurologicalComponentTypes::CONVOLUTIONAL_LAYER;
}

// Multi-Head Masked Self-Attention
MultiHeadMaskedSelfAttentionLayer__BuildInfo::MultiHeadMaskedSelfAttentionLayer__BuildInfo() {
    componentType = cuda_NeurologicalComponentTypes::MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER;
}

// SCC Positional Embedding
SCCPositionalEmbeddingLayer__BuildInfo::SCCPositionalEmbeddingLayer__BuildInfo() {
    componentType = cuda_NeurologicalComponentTypes::SCC_POSITIONAL_EMBEDDING_LAYER;
}

// Chain
ComponentsChain__BuildInfo::ComponentsChain__BuildInfo() {
    componentType = cuda_NeurologicalComponentTypes::COMPONENTS_CHAIN;
}