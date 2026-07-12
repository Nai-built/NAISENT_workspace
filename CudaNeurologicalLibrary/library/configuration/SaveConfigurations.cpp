#include "SaveConfigurations.h"

// Base Component
BaseNeurologicalComponent__SaveInfo::BaseNeurologicalComponent__SaveInfo()
    : componentType(cuda_NeurologicalComponentTypes::DENSE_LAYER) {}

// RMSNorm
RMSNormLayer__SaveInfo::RMSNormLayer__SaveInfo() {
    componentType = cuda_NeurologicalComponentTypes::RMSNORM_LAYER;
}

// Dense
DenseLayer__SaveInfo::DenseLayer__SaveInfo() {
    componentType = cuda_NeurologicalComponentTypes::DENSE_LAYER;
}

// Convolutional
ConvolutionalLayer__SaveInfo::ConvolutionalLayer__SaveInfo() {
    componentType = cuda_NeurologicalComponentTypes::CONVOLUTIONAL_LAYER;
}

// Multi-Head Masked Self-Attention
MultiHeadMaskedSelfAttentionLayer__SaveInfo::MultiHeadMaskedSelfAttentionLayer__SaveInfo() {
    componentType = cuda_NeurologicalComponentTypes::MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER;
}

// SCC Positional Embedding
SCCPositionalEmbeddingLayer__SaveInfo::SCCPositionalEmbeddingLayer__SaveInfo() {
    componentType = cuda_NeurologicalComponentTypes::SCC_POSITIONAL_EMBEDDING_LAYER;
}

// Chain
ComponentsChain__SaveInfo::ComponentsChain__SaveInfo() {
    componentType = cuda_NeurologicalComponentTypes::COMPONENTS_CHAIN;
}
