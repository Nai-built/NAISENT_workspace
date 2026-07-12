#pragma once

#include "../activation_functions/IActivationFunction.h"
#include "../components/INeurologicalComponent.h"
#include "../optimization/IOptimization.h"
#include <vector>
#include <memory>

// ================= Base Optimizer =================
struct BaseOptimizer__BuildInfo {
    cuda_OptimizerTypes optimizerType;

    BaseOptimizer__BuildInfo();
    virtual ~BaseOptimizer__BuildInfo() = default;
};

// ================= ADAM =================
struct ADAM__BuildInfo : public BaseOptimizer__BuildInfo {
    cuda_neurologicalValue beta1;
    cuda_neurologicalValue beta2;
    cuda_neurologicalValue epsilon;

    ADAM__BuildInfo();
};

// ================= Base Component =================
struct BaseNeurologicalComponent__BuildInfo {
    cuda_NeurologicalComponentTypes componentType;
    bool freeze   = false;
    bool residual = false;

    BaseNeurologicalComponent__BuildInfo();
    virtual ~BaseNeurologicalComponent__BuildInfo() = default;
};

// ================= Base Activation =================
struct BaseActivationFunction__BuildInfo {
    cuda_ActivationFunctionTypes componentType;

    BaseActivationFunction__BuildInfo();
    virtual ~BaseActivationFunction__BuildInfo() = default;
};

// ================= ReLU =================
struct ReLU__BuildInfo : public BaseActivationFunction__BuildInfo {
    cuda_neurologicalValue fadeMultiplier;

    ReLU__BuildInfo();
};

// ================= RMSNorm Layer =================
struct RMSNormLayer__BuildInfo : public BaseNeurologicalComponent__BuildInfo {
    int tensorSize;

    RMSNormLayer__BuildInfo();
};
// ================= Dense Layer =================
struct DenseLayer__BuildInfo : public BaseNeurologicalComponent__BuildInfo {
    int inputSize;
    int outputSize;

    int maxDropout;

    std::shared_ptr<BaseActivationFunction__BuildInfo> activation;

    DenseLayer__BuildInfo();
};

// ================= Convolutional Layer =================
struct ConvolutionalLayer__BuildInfo : public BaseNeurologicalComponent__BuildInfo {
    int outputChannels;
    int inputChannels;
    int slideWidth;
    int slideHeight;
    int inputWidth;
    int inputHeight;
    int stride;
    int padding;

    int maxDropout;

    std::shared_ptr<BaseActivationFunction__BuildInfo> activation;

    ConvolutionalLayer__BuildInfo();
};

// ================= Multi-Head Masked Self-Attention =================
struct MultiHeadMaskedSelfAttentionLayer__BuildInfo : public BaseNeurologicalComponent__BuildInfo {
    int inputSize;
    int headOutputSize;
    int headsAmount;

    MultiHeadMaskedSelfAttentionLayer__BuildInfo();
};

// ================= SCC Positional Embedding =================
struct SCCPositionalEmbeddingLayer__BuildInfo : public BaseNeurologicalComponent__BuildInfo {
    int inputSize;
    int outputSize;

    SCCPositionalEmbeddingLayer__BuildInfo();
};

// ================= Chain =================
struct ComponentsChain__BuildInfo : public BaseNeurologicalComponent__BuildInfo {
    std::vector<std::shared_ptr<BaseNeurologicalComponent__BuildInfo>> components;

    ComponentsChain__BuildInfo();
};