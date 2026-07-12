#include "ComponentsChain.h"
#include "DenseLayer.h"
#include "ConvolutionalLayer.h"
#include "RMSNormLayer.h"
#include "MultiHeadMaskedSelfAttentionLayer.h"
#include "SCCPositionalEmbeddingLayer.h"

#include "../cuda/cuda_base.cuh"

// Constructor
cuda_ComponentsChain::cuda_ComponentsChain() = default;

// Static: Initialize all components with offset based on subtree size
void cuda_ComponentsChain::InitializeComponents(cuda_ComponentsChain& chain, unsigned long seed) {
    unsigned long offset = 0;

    for (auto& comp : chain.components) {
        unsigned long compSeed = seed + offset;

        InitializeComponent(comp.get(), compSeed);

        // Increase offset by full subtree size
        offset += CountComponents(comp.get());
    }
}

// Private: Type-dispatch initialization
void cuda_ComponentsChain::InitializeComponent(cuda_INeurologicalComponent* component, unsigned long seed) {
    switch (component->componentType) {
        case cuda_NeurologicalComponentTypes::RMSNORM_LAYER: {
            auto* rms = static_cast<cuda_RMSNormLayer*>(component);

            cuda_RMSNormLayer::Initialize(*rms, seed);
            break;
        }
        case cuda_NeurologicalComponentTypes::DENSE_LAYER: {
            auto denseLayer = static_cast<cuda_DenseLayer*>(component);
            cuda_DenseLayer::Initialize(*denseLayer, seed);
            break;
        }
        case cuda_NeurologicalComponentTypes::CONVOLUTIONAL_LAYER: {
            auto convLayer = static_cast<cuda_ConvolutionalLayer*>(component);
            cuda_ConvolutionalLayer::Initialize(*convLayer, seed);
            break;
        }
        case cuda_NeurologicalComponentTypes::MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER: {
            auto* attn = static_cast<cuda_MultiHeadMaskedSelfAttentionLayer*>(component);
            cuda_MultiHeadMaskedSelfAttentionLayer::Initialize(*attn, seed);
            break;
        }
        case cuda_NeurologicalComponentTypes::SCC_POSITIONAL_EMBEDDING_LAYER: {
            auto* scc = static_cast<cuda_SCCPositionalEmbeddingLayer*>(component);
            cuda_SCCPositionalEmbeddingLayer::Initialize(*scc, seed);
            break;
        }
        case cuda_NeurologicalComponentTypes::COMPONENTS_CHAIN: {
            auto chain = static_cast<cuda_ComponentsChain*>(component);
            cuda_ComponentsChain::InitializeComponents(*chain, seed);
            break;
        }
        default:
            break;
    }
}

// NEW: Count total components (recursive)
unsigned long cuda_ComponentsChain::CountComponents(cuda_INeurologicalComponent* component) {
    switch (component->componentType) {
        case cuda_NeurologicalComponentTypes::DENSE_LAYER:
            return 1;

        case cuda_NeurologicalComponentTypes::CONVOLUTIONAL_LAYER:
            return 1;

        case cuda_NeurologicalComponentTypes::MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER:
            return 1;

        case cuda_NeurologicalComponentTypes::SCC_POSITIONAL_EMBEDDING_LAYER:
            return 1;

        case cuda_NeurologicalComponentTypes::COMPONENTS_CHAIN: {
            auto chain = static_cast<cuda_ComponentsChain*>(component);

            unsigned long count = 0;
            for (auto& comp : chain->components) {
                count += CountComponents(comp.get());
            }
            return count;
        }

        default:
            return 0;
    }
}

// TO DO: 1 - fix reallocation for cuda tensors. [ DONE ] 2 - implement back propagation (ChainAdjustment)

int cuda_ComponentsChain::getOutputRequiredSize(cuda_INeurologicalComponent* comp, const int totalSamples) {
    if (comp->componentType == cuda_NeurologicalComponentTypes::DENSE_LAYER) {
        auto* dense = static_cast<cuda_DenseLayer*>(comp);
        return dense->out_size() * totalSamples;
    }
    else if (comp->componentType == cuda_NeurologicalComponentTypes::CONVOLUTIONAL_LAYER) {
        auto* conv = static_cast<cuda_ConvolutionalLayer*>(comp);
        return conv->out_size() * totalSamples;
    }
    else if (comp->componentType == cuda_NeurologicalComponentTypes::RMSNORM_LAYER) {
        auto* rms = static_cast<cuda_RMSNormLayer*>(comp);
        return rms->size() * totalSamples;
    }
    else if (comp->componentType == cuda_NeurologicalComponentTypes::MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER) {
        auto* attn = static_cast<cuda_MultiHeadMaskedSelfAttentionLayer*>(comp);
        return attn->out_size() * totalSamples;
    }
    else if (comp->componentType == cuda_NeurologicalComponentTypes::SCC_POSITIONAL_EMBEDDING_LAYER) {
        auto* scc = static_cast<cuda_SCCPositionalEmbeddingLayer*>(comp);
        return scc->out_size() * totalSamples;
    }
    else if (comp->componentType == cuda_NeurologicalComponentTypes::COMPONENTS_CHAIN) {
        auto* chain = static_cast<cuda_ComponentsChain*>(comp);
        return cuda_ComponentsChain::getOutputRequiredSize(chain->components[chain->components.size()-1].get(), totalSamples);
    }
}
int cuda_ComponentsChain::getInputRequiredSize(cuda_INeurologicalComponent* comp, const int totalSamples) {
    if (comp->componentType == cuda_NeurologicalComponentTypes::DENSE_LAYER) {
        auto* dense = static_cast<cuda_DenseLayer*>(comp);
        return dense->in_size() * totalSamples;
    }
    else if (comp->componentType == cuda_NeurologicalComponentTypes::CONVOLUTIONAL_LAYER) {
        auto* conv = static_cast<cuda_ConvolutionalLayer*>(comp);
        return conv->in_size() * totalSamples;
    }
    else if (comp->componentType == cuda_NeurologicalComponentTypes::RMSNORM_LAYER) {
        auto* rms = static_cast<cuda_RMSNormLayer*>(comp);
        return rms->size() * totalSamples;
    }
    else if (comp->componentType == cuda_NeurologicalComponentTypes::MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER) {
        auto* attn = static_cast<cuda_MultiHeadMaskedSelfAttentionLayer*>(comp);
        return attn->in_size() * totalSamples;
    }
    else if (comp->componentType == cuda_NeurologicalComponentTypes::SCC_POSITIONAL_EMBEDDING_LAYER) {
        auto* scc = static_cast<cuda_SCCPositionalEmbeddingLayer*>(comp);
        return scc->in_size() * totalSamples;
    }
    else if (comp->componentType == cuda_NeurologicalComponentTypes::COMPONENTS_CHAIN) {
        auto* chain = static_cast<cuda_ComponentsChain*>(comp);
        return cuda_ComponentsChain::getInputRequiredSize(chain->components[0].get(), totalSamples);
    }
}

// Static: Forward activation through the chain
void cuda_ComponentsChain::ChainActivation(
    cuda_ComponentsChain& chain,
    cuda_ConstantTensorSpan input,
    cuda_TensorSpan output,
    const int& batchSize,
    lengths seriesLengths,
    const int& totalSamples
) {
    const int numComponents = static_cast<int>(chain.components.size());

    // Ensure buffer size matches
    if (numComponents > 1) {
        chain.chainActivationBuffers.resize(numComponents - 1);

        for (int i = 0; i < numComponents - 1; ++i) {
            auto* comp = chain.components[i].get();
            int requiredSize = cuda_ComponentsChain::getOutputRequiredSize(comp, totalSamples);

            if (requiredSize > 0 && chain.chainActivationBuffers[i].size < requiredSize) {
                chain.chainActivationBuffers[i] = cuda_Tensor(requiredSize);
            }
            chain.chainActivationBuffers[i].zeroAll();
        }
    }

    // Forward pass
    for (int i = 0; i < numComponents; ++i) {
        auto* comp = chain.components[i].get();

        cuda_ConstantTensorSpan currentInput;
        cuda_TensorSpan currentOutput;

        // Input selection
        if (i == 0) {
            currentInput = input;
        } else {
            currentInput = cuda_ConstantTensorSpan(chain.chainActivationBuffers[i - 1]);
        }

        // Output selection
        if (i == numComponents - 1) {
            currentOutput = output;
        } else {
            currentOutput = cuda_TensorSpan(chain.chainActivationBuffers[i]);
        }

        // printf("forward: %i\n", i+1);

        switch (comp->componentType) {
            case cuda_NeurologicalComponentTypes::RMSNORM_LAYER: {
                auto* rms = static_cast<cuda_RMSNormLayer*>(comp);

                cuda_RMSNormLayer::ForwardPass(
                    *rms,
                    currentInput,
                    currentOutput,
                    totalSamples
                );

                break;
            }
            case cuda_NeurologicalComponentTypes::DENSE_LAYER: {
                auto* dense = static_cast<cuda_DenseLayer*>(comp);

                // printf("dense: %i x %i\n", dense->in_size(), dense->out_size());

                cuda_DenseLayer::ForwardPass(
                    *dense,
                    currentInput,
                    currentOutput,
                    totalSamples
                );

                break;
            }
            case cuda_NeurologicalComponentTypes::CONVOLUTIONAL_LAYER: {
                auto* conv = static_cast<cuda_ConvolutionalLayer*>(comp);

                cuda_ConvolutionalLayer::ForwardPass(
                    *conv,
                    currentInput,
                    currentOutput,
                    totalSamples
                );

                break;
            }

            case cuda_NeurologicalComponentTypes::MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER: {
                auto* attn = static_cast<cuda_MultiHeadMaskedSelfAttentionLayer*>(comp);

                cuda_MultiHeadMaskedSelfAttentionLayer::ForwardPass(
                    *attn,
                    currentInput,
                    currentOutput,
                    batchSize,
                    seriesLengths,
                    totalSamples
                );

                break;
            }

            case cuda_NeurologicalComponentTypes::SCC_POSITIONAL_EMBEDDING_LAYER: {
                auto* scc = static_cast<cuda_SCCPositionalEmbeddingLayer*>(comp);

                cuda_SCCPositionalEmbeddingLayer::ForwardPass(
                    *scc,
                    currentInput,
                    currentOutput,
                    batchSize,
                    seriesLengths,
                    totalSamples
                );

                break;
            }

            case cuda_NeurologicalComponentTypes::COMPONENTS_CHAIN: {
                auto* subChain = static_cast<cuda_ComponentsChain*>(comp);

                // printf("chain activation\n");

                cuda_ComponentsChain::ChainActivation(
                    *subChain,
                    currentInput,
                    currentOutput,
                    batchSize,
                    seriesLengths,
                    totalSamples
                );

                break;
            }

            default:
                printf("NOOO\n");
                break;
        }

        if (comp->residual) {
            addition(currentOutput.data(), currentInput.data()
            , currentOutput.size, currentInput.size);
        }
    }
}
// Static: Inference through the chain
void cuda_ComponentsChain::ChainInference(
    cuda_ComponentsChain& chain,
    cuda_ConstantTensorSpan input,
    cuda_TensorSpan output,
    const int& sequenceLength
) {
    const int numComponents = static_cast<int>(chain.components.size());

    if (numComponents > 1) {
        chain.chainActivationBuffers.resize(numComponents - 1);

        for (int i = 0; i < numComponents - 1; ++i) {
            auto* comp = chain.components[i].get();
            int requiredSize = cuda_ComponentsChain::getOutputRequiredSize(comp, sequenceLength);

            if (requiredSize > 0 && chain.chainActivationBuffers[i].size < requiredSize) {
                chain.chainActivationBuffers[i] = cuda_Tensor(requiredSize);
            }
            chain.chainActivationBuffers[i].zeroAll();
        }
    }

    for (int i = 0; i < numComponents; ++i) {
        auto* comp = chain.components[i].get();

        cuda_ConstantTensorSpan currentInput;
        cuda_TensorSpan currentOutput;

        if (i == 0) {
            currentInput = input;
        } else {
            currentInput = cuda_ConstantTensorSpan(chain.chainActivationBuffers[i - 1]);
        }

        if (i == numComponents - 1) {
            currentOutput = output;
        } else {
            currentOutput = cuda_TensorSpan(chain.chainActivationBuffers[i]);
        }

        switch (comp->componentType) {
            case cuda_NeurologicalComponentTypes::RMSNORM_LAYER: {
                auto* rms = static_cast<cuda_RMSNormLayer*>(comp);
                cuda_RMSNormLayer::Infer(*rms, currentInput, currentOutput, sequenceLength);
                break;
            }
            case cuda_NeurologicalComponentTypes::DENSE_LAYER: {
                auto* dense = static_cast<cuda_DenseLayer*>(comp);
                cuda_DenseLayer::Infer(*dense, currentInput, currentOutput, sequenceLength);
                break;
            }
            case cuda_NeurologicalComponentTypes::CONVOLUTIONAL_LAYER: {
                auto* conv = static_cast<cuda_ConvolutionalLayer*>(comp);
                cuda_ConvolutionalLayer::Infer(*conv, currentInput, currentOutput, sequenceLength);
                break;
            }
            case cuda_NeurologicalComponentTypes::MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER: {
                auto* attn = static_cast<cuda_MultiHeadMaskedSelfAttentionLayer*>(comp);
                cuda_MultiHeadMaskedSelfAttentionLayer::Infer(*attn, currentInput, currentOutput, sequenceLength);
                break;
            }
            case cuda_NeurologicalComponentTypes::SCC_POSITIONAL_EMBEDDING_LAYER: {
                auto* scc = static_cast<cuda_SCCPositionalEmbeddingLayer*>(comp);
                cuda_SCCPositionalEmbeddingLayer::Infer(*scc, currentInput, currentOutput, sequenceLength);
                break;
            }
            case cuda_NeurologicalComponentTypes::COMPONENTS_CHAIN: {
                auto* subChain = static_cast<cuda_ComponentsChain*>(comp);
                cuda_ComponentsChain::ChainInference(*subChain, currentInput, currentOutput, sequenceLength);
                break;
            }
            default:
                break;
        }

        if (comp->residual) {
            addition(currentOutput.data(), currentInput.data(), currentOutput.size, currentInput.size);
        }
    }
}
void cuda_ComponentsChain::ClearInference(cuda_ComponentsChain& chain) {
    const int numComponents = static_cast<int>(chain.components.size());

    for (int i = 0; i < numComponents; ++i) {
        auto* comp = chain.components[i].get();

        if (comp->freeze) continue;

        switch (comp->componentType) {
            case cuda_NeurologicalComponentTypes::RMSNORM_LAYER: {
                auto* rms = static_cast<cuda_RMSNormLayer*>(comp);

                cuda_RMSNormLayer::ClearInference(*rms);
                break;
            }
            case cuda_NeurologicalComponentTypes::DENSE_LAYER: {
                auto* dense = static_cast<cuda_DenseLayer*>(comp);

                cuda_DenseLayer::ClearInference(*dense);
                break;
            }
            case cuda_NeurologicalComponentTypes::CONVOLUTIONAL_LAYER: {
                auto* conv = static_cast<cuda_ConvolutionalLayer*>(comp);

                cuda_ConvolutionalLayer::ClearInference(*conv);
                break;
            }

            case cuda_NeurologicalComponentTypes::MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER: {
                auto* attn = static_cast<cuda_MultiHeadMaskedSelfAttentionLayer*>(comp);

                cuda_MultiHeadMaskedSelfAttentionLayer::ClearInference(*attn);
                break;
            }

            case cuda_NeurologicalComponentTypes::SCC_POSITIONAL_EMBEDDING_LAYER: {
                auto* scc = static_cast<cuda_SCCPositionalEmbeddingLayer*>(comp);

                cuda_SCCPositionalEmbeddingLayer::ClearInference(*scc);
                break;
            }

            case cuda_NeurologicalComponentTypes::COMPONENTS_CHAIN: {
                auto* subChain = static_cast<cuda_ComponentsChain*>(comp);

                cuda_ComponentsChain::ClearInference(
                    *subChain
                );
                break;
            }

            default:
                break;
        }
    }
}

// Static: Backward propagation through the chain
void cuda_ComponentsChain::ChainAdjustment(
    cuda_ComponentsChain& chain,
    cuda_ConstantTensorSpan output_gradient,
    cuda_TensorSpan input_gradient,
    const int& batchSize,
    lengths seriesLengths,
    const int& totalSamples
) {
    const int numComponents = static_cast<int>(chain.components.size());

    // Ensure buffer size matches
    if (numComponents > 1) {
        chain.chainGradientBuffers.resize(numComponents - 1);

        for (int i = 1; i < numComponents; ++i) {
            auto* comp = chain.components[i].get();
            int requiredSize = cuda_ComponentsChain::getInputRequiredSize(comp, totalSamples);

            if (requiredSize > 0 && chain.chainGradientBuffers[i - 1].size < requiredSize) {
                chain.chainGradientBuffers[i - 1] = cuda_Tensor(requiredSize);
            }
            chain.chainGradientBuffers[i - 1].zeroAll();
        }
    }

    // Backward pass (reverse order)
    for (int i = numComponents - 1; i >= 0; --i) {
        auto* comp = chain.components[i].get();

        cuda_ConstantTensorSpan currentOutputGrad;
        cuda_TensorSpan currentInputGrad;

        // Output gradient selection
        if (i == numComponents - 1) {
            currentOutputGrad = output_gradient;
        } else {
            currentOutputGrad = cuda_ConstantTensorSpan(chain.chainGradientBuffers[i]);
        }

        // Input gradient selection
        if (i == 0) {
            currentInputGrad = input_gradient;
        } else {
            currentInputGrad = cuda_TensorSpan(chain.chainGradientBuffers[i - 1]);
        }

        // printf("backward: %i\n", i+1);

        switch (comp->componentType) {
            case cuda_NeurologicalComponentTypes::RMSNORM_LAYER: {
                auto* rms = static_cast<cuda_RMSNormLayer*>(comp);

                cuda_RMSNormLayer::BackPropagation(
                    *rms,
                    currentOutputGrad,
                    currentInputGrad,
                    totalSamples
                );

                break;
            }
            case cuda_NeurologicalComponentTypes::DENSE_LAYER: {
                auto* dense = static_cast<cuda_DenseLayer*>(comp);

                cuda_DenseLayer::BackPropagation(
                    *dense,
                    currentOutputGrad,
                    currentInputGrad,
                    totalSamples
                );

                break;
            }
            case cuda_NeurologicalComponentTypes::CONVOLUTIONAL_LAYER: {
                auto* conv = static_cast<cuda_ConvolutionalLayer*>(comp);

                cuda_ConvolutionalLayer::BackPropagation(
                    *conv,
                    currentOutputGrad,
                    currentInputGrad,
                    totalSamples
                );

                break;
            }

            case cuda_NeurologicalComponentTypes::MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER: {
                auto* attn = static_cast<cuda_MultiHeadMaskedSelfAttentionLayer*>(comp);

                cuda_MultiHeadMaskedSelfAttentionLayer::BackPropagation(
                    *attn,
                    currentOutputGrad,
                    currentInputGrad,
                    batchSize,
                    seriesLengths,
                    totalSamples
                );

                break;
            }

            case cuda_NeurologicalComponentTypes::SCC_POSITIONAL_EMBEDDING_LAYER: {
                auto* scc = static_cast<cuda_SCCPositionalEmbeddingLayer*>(comp);

                cuda_SCCPositionalEmbeddingLayer::BackPropagation(
                    *scc,
                    currentOutputGrad,
                    currentInputGrad,
                    batchSize,
                    seriesLengths,
                    totalSamples
                );

                break;
            }

            case cuda_NeurologicalComponentTypes::COMPONENTS_CHAIN: {
                auto* subChain = static_cast<cuda_ComponentsChain*>(comp);

                cuda_ComponentsChain::ChainAdjustment(
                    *subChain,
                    currentOutputGrad,
                    currentInputGrad,
                    batchSize,
                    seriesLengths,
                    totalSamples
                );

                break;
            }

            default:
                printf("NOOO\n");
                break;
        }

        if (comp->residual) {
            addition(currentInputGrad.data(), currentOutputGrad.data()
            , currentInputGrad.size, currentOutputGrad.size);
        }
    }
}
// Static: Apply updates across the chain
void cuda_ComponentsChain::ChainUpdate(
    cuda_ComponentsChain& chain,
    const cuda_neurologicalValue& updateMultiplier
) {
    const int numComponents = static_cast<int>(chain.components.size());

    for (int i = 0; i < numComponents; ++i) {
        auto* comp = chain.components[i].get();

        if (comp->freeze) continue;

        switch (comp->componentType) {
            case cuda_NeurologicalComponentTypes::RMSNORM_LAYER: {
                auto* rms = static_cast<cuda_RMSNormLayer*>(comp);

                cuda_RMSNormLayer::Update(*rms, updateMultiplier);
                break;
            }
            case cuda_NeurologicalComponentTypes::DENSE_LAYER: {
                auto* dense = static_cast<cuda_DenseLayer*>(comp);

                cuda_DenseLayer::Update(*dense, updateMultiplier);
                break;
            }
            case cuda_NeurologicalComponentTypes::CONVOLUTIONAL_LAYER: {
                auto* conv = static_cast<cuda_ConvolutionalLayer*>(comp);

                cuda_ConvolutionalLayer::Update(*conv, updateMultiplier);
                break;
            }

            case cuda_NeurologicalComponentTypes::MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER: {
                auto* attn = static_cast<cuda_MultiHeadMaskedSelfAttentionLayer*>(comp);

                cuda_MultiHeadMaskedSelfAttentionLayer::Update(*attn, updateMultiplier);
                break;
            }

            case cuda_NeurologicalComponentTypes::SCC_POSITIONAL_EMBEDDING_LAYER: {
                auto* scc = static_cast<cuda_SCCPositionalEmbeddingLayer*>(comp);

                cuda_SCCPositionalEmbeddingLayer::Update(*scc, updateMultiplier);
                break;
            }

            case cuda_NeurologicalComponentTypes::COMPONENTS_CHAIN: {
                auto* subChain = static_cast<cuda_ComponentsChain*>(comp);

                cuda_ComponentsChain::ChainUpdate(
                    *subChain,
                    updateMultiplier
                );
                break;
            }

            default:
                break;
        }
    }
}