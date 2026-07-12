#pragma once

#include "../components/INeurologicalComponent.h"
#include <vector>
#include <memory>

// ================= Base Component =================
struct BaseNeurologicalComponent__SaveInfo {
    cuda_NeurologicalComponentTypes componentType;

    BaseNeurologicalComponent__SaveInfo();
    virtual ~BaseNeurologicalComponent__SaveInfo() = default;
};

// ================= RMSNorm Layer =================
struct RMSNormLayer__SaveInfo : public BaseNeurologicalComponent__SaveInfo {
    std::vector<cuda_neurologicalValue> gamma;   // size = tensorSize

    RMSNormLayer__SaveInfo();
};

// ================= Dense Layer =================
struct DenseLayer__SaveInfo : public BaseNeurologicalComponent__SaveInfo {
    std::vector<cuda_neurologicalValue> weights;   // size = inputSize * outputSize  → layout: [in][out]
    std::vector<cuda_neurologicalValue> biases;    // size = outputSize

    DenseLayer__SaveInfo();
};

// ================= Convolutional Layer =================
struct ConvolutionalLayer__SaveInfo : public BaseNeurologicalComponent__SaveInfo {
    std::vector<cuda_neurologicalValue> weights;   // size = outputChannels * inputChannels * slideWidth * slideHeight
    std::vector<cuda_neurologicalValue> biases;    // size = outputChannels

    ConvolutionalLayer__SaveInfo();
};

// ================= Multi-Head Masked Self-Attention =================
struct MultiHeadMaskedSelfAttentionLayer__SaveInfo : public BaseNeurologicalComponent__SaveInfo {
    std::vector<cuda_neurologicalValue> weights;   // size = inputSize * headsAmount * headOutputSize * 3  (K, V, Q)

    MultiHeadMaskedSelfAttentionLayer__SaveInfo();
};

// ================= SCC Positional Embedding =================
struct SCCPositionalEmbeddingLayer__SaveInfo : public BaseNeurologicalComponent__SaveInfo {
    std::vector<cuda_neurologicalValue> weights;   // size = inputSize * outputSize  → layout: [in][out]

    SCCPositionalEmbeddingLayer__SaveInfo();
};

// ================= Chain =================
struct ComponentsChain__SaveInfo : public BaseNeurologicalComponent__SaveInfo {
    std::vector<std::shared_ptr<BaseNeurologicalComponent__SaveInfo>> components;

    ComponentsChain__SaveInfo();
};
