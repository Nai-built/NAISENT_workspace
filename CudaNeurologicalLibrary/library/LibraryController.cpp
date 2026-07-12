#include <iostream>
#include <unordered_map>
#include <string>

#include "activation_functions/ReLU.h"

#include "optimization/ADAM_optimization.h"

#include "components/RMSNormLayer.h"
#include "components/DenseLayer.h"
#include "components/ConvolutionalLayer.h"
#include "components/MultiHeadMaskedSelfAttentionLayer.h"
#include "components/SCCPositionalEmbeddingLayer.h"
#include "components/ComponentsChain.h"
#include "components/INeurologicalComponent.h"

#include "LossFunctions.h"

#include "configuration/SaveConfigurations.h"
#include "LibraryController.h"

#include <iostream>

inline lengths bufferToConstantLengths(pybind11::buffer buffer) {
    pybind11::buffer_info bufferInfo = buffer.request();

    if (bufferInfo.ndim != 1)
        throw std::runtime_error("Expected 1D buffer");

    if (bufferInfo.format != pybind11::format_descriptor<int>::format())
        throw std::runtime_error("Expected int value buffer");

    if (bufferInfo.strides[0] != sizeof(int))
        throw std::runtime_error("Buffer is not contiguous");

    int* bufferData = static_cast<int*>(bufferInfo.ptr);
    size_t bufferSize = static_cast<size_t>(bufferInfo.shape[0]);

    return lengths(bufferData, bufferSize);
}

inline neurologicalConstantSpan bufferToConstantSpan(pybind11::buffer buffer) {
    pybind11::buffer_info bufferInfo = buffer.request();

    if (bufferInfo.ndim != 1)
        throw std::runtime_error("Expected 1D buffer");

    if (bufferInfo.format != pybind11::format_descriptor<cuda_neurologicalValue>::format())
        throw std::runtime_error("Expected neurological value buffer");

    if (bufferInfo.strides[0] != sizeof(cuda_neurologicalValue))
        throw std::runtime_error("Buffer is not contiguous");

    cuda_neurologicalValue* bufferData = static_cast<cuda_neurologicalValue*>(bufferInfo.ptr);
    size_t bufferSize = static_cast<size_t>(bufferInfo.shape[0]);

    return neurologicalConstantSpan(bufferData, bufferSize);
}
inline neurologicalSpan bufferToSpan(pybind11::buffer buffer) {
    pybind11::buffer_info bufferInfo = buffer.request();

    if (bufferInfo.readonly)
        throw std::runtime_error("Buffer is read-only");

    if (bufferInfo.ndim != 1)
        throw std::runtime_error("Expected 1D buffer");

    if (bufferInfo.format != pybind11::format_descriptor<cuda_neurologicalValue>::format())
        throw std::runtime_error("Expected neurological value buffer");

    if (bufferInfo.strides[0] != sizeof(cuda_neurologicalValue))
        throw std::runtime_error("Buffer is not contiguous");

    cuda_neurologicalValue* bufferData = static_cast<cuda_neurologicalValue*>(bufferInfo.ptr);
    size_t bufferSize = static_cast<size_t>(bufferInfo.shape[0]);

    return neurologicalSpan(bufferData, bufferSize);
}

static int g_ModelCounter = 0;

std::unordered_map<std::string, std::unique_ptr<cuda_ComponentsChain>> g_BuiltModels;

// Forward declaration
static std::unique_ptr<cuda_INeurologicalComponent>
BuildComponent(const BaseNeurologicalComponent__BuildInfo& info, const BaseOptimizer__BuildInfo& optInfo);
static std::unique_ptr<cuda_IActivationFunction>
BuildActivation(const BaseActivationFunction__BuildInfo& info);
static std::unique_ptr<cuda_IOptimization>
BuildOptimization(const BaseOptimizer__BuildInfo& info, const int& parameterSize);

// Build full model (entry point)
std::string BuildModel(const ComponentsChain__BuildInfo& buildInfo, const BaseOptimizer__BuildInfo& optInfo) {
    auto chain = std::make_unique<cuda_ComponentsChain>();

    for (const auto& compInfo : buildInfo.components) {
        chain->components.push_back(
            BuildComponent(*compInfo, optInfo)
        );
    }

    std::string modelId = "model_" + std::to_string(g_ModelCounter++);
    g_BuiltModels[modelId] = std::move(chain);

    return modelId;
}

// Recursive builder
static std::unique_ptr<cuda_INeurologicalComponent>
BuildComponent(const BaseNeurologicalComponent__BuildInfo& info, const BaseOptimizer__BuildInfo& optInfo) {
    switch (info.componentType) {
        case cuda_NeurologicalComponentTypes::RMSNORM_LAYER: {
            const auto& rmsInfo = static_cast<const RMSNormLayer__BuildInfo&>(info);

            auto layer = std::make_unique<cuda_RMSNormLayer>(
                rmsInfo.tensorSize
            );

            layer->componentType = cuda_NeurologicalComponentTypes::RMSNORM_LAYER;
            layer->freeze   = rmsInfo.freeze;
            layer->residual = rmsInfo.residual;
            if (optInfo.optimizerType != cuda_OptimizerTypes::DEFAULT) {
                layer->gamma_optimization = BuildOptimization(optInfo, layer->gamma.size);
            }

            return layer;
        }
        case cuda_NeurologicalComponentTypes::DENSE_LAYER: {
            const auto& denseInfo = static_cast<const DenseLayer__BuildInfo&>(info);

            auto layer = std::make_unique<cuda_DenseLayer>(
                denseInfo.inputSize,
                denseInfo.outputSize,

                denseInfo.maxDropout
            );

            layer->componentType = cuda_NeurologicalComponentTypes::DENSE_LAYER;
            layer->freeze   = denseInfo.freeze;
            layer->residual = denseInfo.residual;

            // Build activation if exists
            if (denseInfo.activation) {
                layer->activation = BuildActivation(*denseInfo.activation);
            }
            if (optInfo.optimizerType != cuda_OptimizerTypes::DEFAULT) {
                layer->weights_optimization = BuildOptimization(optInfo, layer->weights.size);
                layer->biases_optimization = BuildOptimization(optInfo, layer->biases.size);
            }
            return layer;
        }

        case cuda_NeurologicalComponentTypes::CONVOLUTIONAL_LAYER: {
            const auto& convInfo = static_cast<const ConvolutionalLayer__BuildInfo&>(info);

            auto layer = std::make_unique<cuda_ConvolutionalLayer>(
                convInfo.outputChannels,
                convInfo.inputChannels,
                convInfo.slideWidth,
                convInfo.slideHeight,
                convInfo.inputWidth,
                convInfo.inputHeight,
                convInfo.stride,
                convInfo.padding,

                convInfo.maxDropout
            );

            layer->componentType = cuda_NeurologicalComponentTypes::CONVOLUTIONAL_LAYER;
            layer->freeze   = convInfo.freeze;
            layer->residual = convInfo.residual;

            // Build activation if exists
            if (convInfo.activation) {
                layer->activation = BuildActivation(*convInfo.activation);
            }
            if (optInfo.optimizerType != cuda_OptimizerTypes::DEFAULT) {
                layer->weights_optimization = BuildOptimization(optInfo, layer->weights.size);
                layer->biases_optimization = BuildOptimization(optInfo, layer->biases.size);
            }
            return layer;
        }

        case cuda_NeurologicalComponentTypes::MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER: {
            const auto& attnInfo = static_cast<const MultiHeadMaskedSelfAttentionLayer__BuildInfo&>(info);

            auto layer = std::make_unique<cuda_MultiHeadMaskedSelfAttentionLayer>(
                attnInfo.inputSize,
                attnInfo.headOutputSize,
                attnInfo.headsAmount
            );
            layer->componentType = cuda_NeurologicalComponentTypes::MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER;
            layer->freeze   = attnInfo.freeze;
            layer->residual = attnInfo.residual;

            if (optInfo.optimizerType != cuda_OptimizerTypes::DEFAULT) {
                layer->weights_optimization = BuildOptimization(optInfo, layer->weights.size);
            }

            return layer;
        }

        case cuda_NeurologicalComponentTypes::SCC_POSITIONAL_EMBEDDING_LAYER: {
            const auto& sccInfo = static_cast<const SCCPositionalEmbeddingLayer__BuildInfo&>(info);

            auto layer = std::make_unique<cuda_SCCPositionalEmbeddingLayer>(
                sccInfo.inputSize,
                sccInfo.outputSize
            );
            layer->componentType = cuda_NeurologicalComponentTypes::SCC_POSITIONAL_EMBEDDING_LAYER;
            layer->freeze   = sccInfo.freeze;
            layer->residual = sccInfo.residual;

            if (optInfo.optimizerType != cuda_OptimizerTypes::DEFAULT) {
                layer->weights_optimization = BuildOptimization(optInfo, layer->weights.size);
            }

            return layer;
        }

        case cuda_NeurologicalComponentTypes::COMPONENTS_CHAIN: {
            const auto& chainInfo = static_cast<const ComponentsChain__BuildInfo&>(info);

            auto chain = std::make_unique<cuda_ComponentsChain>();
            chain->componentType = cuda_NeurologicalComponentTypes::COMPONENTS_CHAIN;
            chain->freeze   = chainInfo.freeze;
            chain->residual = chainInfo.residual;

            for (const auto& subComp : chainInfo.components) {
                chain->components.push_back(
                    BuildComponent(*subComp, optInfo)
                );
            }

            return chain;
        }

        default:
            return nullptr;
    }
}
// Build Activation
static std::unique_ptr<cuda_IActivationFunction>
BuildActivation(const BaseActivationFunction__BuildInfo& info) {
    switch (info.componentType) {

        case cuda_ActivationFunctionTypes::ReLU: {
            const auto& reluInfo = static_cast<const ReLU__BuildInfo&>(info);

            auto relu = std::make_unique<cuda_ReLU>();
            relu->activationType = cuda_ActivationFunctionTypes::ReLU;
            relu->fadeMultiplier = reluInfo.fadeMultiplier;

            return relu;
        }

        default:
            return nullptr;
    }
}
// void testActivation(const cuda_neurologicalConstantSpan input, const cuda_neurologicalSpan output, const int& batchSize) {
//     std::cout << "CUDA" << std::endl;
// }

// Build Optimizer
static std::unique_ptr<cuda_IOptimization>
BuildOptimization(const BaseOptimizer__BuildInfo& info, const int& parameterSize) {
    switch (info.optimizerType) {

        case cuda_OptimizerTypes::ADAM: {
            const auto& adamInfo = static_cast<const ADAM__BuildInfo&>(info);

            auto opt = std::make_unique<cuda_ADAM_optimization>();

            opt->optimizerType = cuda_OptimizerTypes::ADAM;
            opt->t = 0;

            opt->beta1 = adamInfo.beta1;
            opt->beta2 = adamInfo.beta2;
            opt->epsilon = adamInfo.epsilon;

            opt->m = cuda_Tensor(parameterSize);
            opt->v = cuda_Tensor(parameterSize);

            return opt;
        }

        default:
            return nullptr;
    }
}

std::unordered_map<std::string, cuda_Tensor> g_ModelInputCache;

// Forward declaration
static std::shared_ptr<BaseNeurologicalComponent__SaveInfo>
ExtractComponent(cuda_INeurologicalComponent* component);

// Recursive extractor
static std::shared_ptr<BaseNeurologicalComponent__SaveInfo>
ExtractComponent(cuda_INeurologicalComponent* component) {
    switch (component->componentType) {
        case cuda_NeurologicalComponentTypes::RMSNORM_LAYER: {
            printf("extracting NORM\n");

            cuda_RMSNormLayer* layer = static_cast<cuda_RMSNormLayer*>(component);
            std::shared_ptr<RMSNormLayer__SaveInfo> saveInfo = std::make_shared<RMSNormLayer__SaveInfo>();
            saveInfo->componentType = cuda_NeurologicalComponentTypes::RMSNORM_LAYER;
            
            saveInfo->gamma.resize(layer->gamma.size);
            cuda_Tensor::toCPU(layer->gamma, neurologicalSpan(saveInfo->gamma));

            printf("extracted NORM\n");
            return saveInfo;
        }
        case cuda_NeurologicalComponentTypes::DENSE_LAYER: {
            printf("extracting DENSE\n");

            cuda_DenseLayer* layer = static_cast<cuda_DenseLayer*>(component);
            std::shared_ptr<DenseLayer__SaveInfo> saveInfo = std::make_shared<DenseLayer__SaveInfo>();
            saveInfo->componentType = cuda_NeurologicalComponentTypes::DENSE_LAYER;

            saveInfo->weights.resize(layer->weights.size);
            saveInfo->biases.resize(layer->biases.size);
            cuda_Tensor::toCPU(layer->weights, neurologicalSpan(saveInfo->weights));
            cuda_Tensor::toCPU(layer->biases, neurologicalSpan(saveInfo->biases));

            printf("extracted DENSE\n");
            return saveInfo;
        }
        case cuda_NeurologicalComponentTypes::CONVOLUTIONAL_LAYER: {
            printf("extracting CONV\n");

            cuda_ConvolutionalLayer* layer = static_cast<cuda_ConvolutionalLayer*>(component);
            std::shared_ptr<ConvolutionalLayer__SaveInfo> saveInfo = std::make_shared<ConvolutionalLayer__SaveInfo>();
            saveInfo->componentType = cuda_NeurologicalComponentTypes::CONVOLUTIONAL_LAYER;

            saveInfo->weights.resize(layer->weights.size);
            saveInfo->biases.resize(layer->biases.size);
            cuda_Tensor::toCPU(layer->weights, neurologicalSpan(saveInfo->weights));
            cuda_Tensor::toCPU(layer->biases, neurologicalSpan(saveInfo->biases));

            printf("extracted CONV\n");
            return saveInfo;
        }
        case cuda_NeurologicalComponentTypes::MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER: {
            printf("extracting ATTENTION\n");

            cuda_MultiHeadMaskedSelfAttentionLayer* layer =
                static_cast<cuda_MultiHeadMaskedSelfAttentionLayer*>(component);
            std::shared_ptr<MultiHeadMaskedSelfAttentionLayer__SaveInfo> saveInfo = std::make_shared<MultiHeadMaskedSelfAttentionLayer__SaveInfo>();
            saveInfo->componentType = cuda_NeurologicalComponentTypes::MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER;
            
            saveInfo->weights.resize(layer->weights.size);
            cuda_Tensor::toCPU(layer->weights, neurologicalSpan(saveInfo->weights));

            printf("extracted ATTENTION\n");
            return saveInfo;
        }
        case cuda_NeurologicalComponentTypes::SCC_POSITIONAL_EMBEDDING_LAYER: {
            printf("extracting POSITIONAL\n");

            cuda_SCCPositionalEmbeddingLayer* layer =
                static_cast<cuda_SCCPositionalEmbeddingLayer*>(component);
            std::shared_ptr<SCCPositionalEmbeddingLayer__SaveInfo> saveInfo = std::make_shared<SCCPositionalEmbeddingLayer__SaveInfo>();
            saveInfo->componentType = cuda_NeurologicalComponentTypes::SCC_POSITIONAL_EMBEDDING_LAYER;
            
            saveInfo->weights.resize(layer->weights.size);
            cuda_Tensor::toCPU(layer->weights, neurologicalSpan(saveInfo->weights));

            printf("extracted POSITIONAL\n");
            return saveInfo;
        }
        case cuda_NeurologicalComponentTypes::COMPONENTS_CHAIN: {
            printf("extracting CHAIN\n");

            cuda_ComponentsChain* chain = static_cast<cuda_ComponentsChain*>(component);
            std::shared_ptr<ComponentsChain__SaveInfo> saveInfo = std::make_shared<ComponentsChain__SaveInfo>();
            saveInfo->componentType = cuda_NeurologicalComponentTypes::COMPONENTS_CHAIN;
            
            for (const auto& comp : chain->components) {
                saveInfo->components.push_back(ExtractComponent(comp.get()));
            }

            printf("extracted CHAIN\n");
            return saveInfo;
        }
        default:
            return nullptr;
    }
}

// Extract model parameters to CPU
std::shared_ptr<BaseNeurologicalComponent__SaveInfo> ExtractModel(const std::string& modelId) {
    auto& model = g_BuiltModels[modelId];
    printf("EXTRACTING FROM: %s\n", modelId);
    return ExtractComponent(model.get());
}

// Forward declaration
static bool InsertComponent(cuda_INeurologicalComponent* component,
                            const BaseNeurologicalComponent__SaveInfo& saveInfo);

// Recursive inserter
static bool InsertComponent(cuda_INeurologicalComponent* component,
                            const BaseNeurologicalComponent__SaveInfo& saveInfo) {
    switch (component->componentType) {
        case cuda_NeurologicalComponentTypes::RMSNORM_LAYER: {
            printf("inserting NORM\n");

            cuda_RMSNormLayer* layer = static_cast<cuda_RMSNormLayer*>(component);
            const auto& s = static_cast<const RMSNormLayer__SaveInfo&>(saveInfo);
            layer->gamma = cuda_Tensor(neurologicalConstantSpan(s.gamma));
            
            printf("inserted NORM\n");

            return true;
        }
        case cuda_NeurologicalComponentTypes::DENSE_LAYER: {
            printf("inserting DENSE\n");

            cuda_DenseLayer* layer = static_cast<cuda_DenseLayer*>(component);
            const auto& s = static_cast<const DenseLayer__SaveInfo&>(saveInfo);
            layer->weights = cuda_Tensor(neurologicalConstantSpan(s.weights));
            layer->biases  = cuda_Tensor(neurologicalConstantSpan(s.biases));
            
            printf("inserted DENSE\n");

            return true;
        }
        case cuda_NeurologicalComponentTypes::CONVOLUTIONAL_LAYER: {
            printf("inserting CONV\n");

            cuda_ConvolutionalLayer* layer = static_cast<cuda_ConvolutionalLayer*>(component);
            const auto& s = static_cast<const ConvolutionalLayer__SaveInfo&>(saveInfo);
            layer->weights = cuda_Tensor(neurologicalConstantSpan(s.weights));
            layer->biases  = cuda_Tensor(neurologicalConstantSpan(s.biases));

            printf("inserted CONV\n");

            return true;
        }
        case cuda_NeurologicalComponentTypes::MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER: {
            printf("inserting ATTENTION\n");

            cuda_MultiHeadMaskedSelfAttentionLayer* layer =
                static_cast<cuda_MultiHeadMaskedSelfAttentionLayer*>(component);
            const auto& s = static_cast<const MultiHeadMaskedSelfAttentionLayer__SaveInfo&>(saveInfo);
            layer->weights = cuda_Tensor(neurologicalConstantSpan(s.weights));
            
            printf("inserted ATTENTION\n");

            return true;
        }
        case cuda_NeurologicalComponentTypes::SCC_POSITIONAL_EMBEDDING_LAYER: {
            printf("inserting POSITIONAL\n");

            cuda_SCCPositionalEmbeddingLayer* layer =
                static_cast<cuda_SCCPositionalEmbeddingLayer*>(component);
            const auto& s = static_cast<const SCCPositionalEmbeddingLayer__SaveInfo&>(saveInfo);
            layer->weights = cuda_Tensor(neurologicalConstantSpan(s.weights));
            
            printf("inserted POSITIONAL\n");

            return true;
        }
        case cuda_NeurologicalComponentTypes::COMPONENTS_CHAIN: {
            printf("inserting CHAIN\n");

            cuda_ComponentsChain* chain = static_cast<cuda_ComponentsChain*>(component);
            const auto& s = static_cast<const ComponentsChain__SaveInfo&>(saveInfo);
            for (int i = 0; i < chain->components.size(); ++i) {
                if (i >= (int)s.components.size()) break;
                InsertComponent(chain->components[i].get(), *s.components[i]);
            }
            
            printf("inserted CHAIN\n");

            return true;
        }
        default:
            printf("default? %i\n", component->componentType);

            return false;
    }

    return true;
}

// Insert model parameters from CPU
bool InsertModel(const std::string& modelId,
                 const std::shared_ptr<BaseNeurologicalComponent__SaveInfo>& saveInfo) {
    auto& model = g_BuiltModels[modelId];
    printf("INSERTING TO: %s\n", modelId);
    return InsertComponent(model.get(), *saveInfo);
}

// Initialize
void InitializeModel(const std::string& modelId, unsigned long seed) {
    auto& model = g_BuiltModels[modelId];
    cuda_ComponentsChain::InitializeComponents(*model, seed);
}

// Infer
void InferModel(
    const std::string& modelId,
    pybind11::buffer inputBuffer,
    pybind11::buffer outputBuffer,
    const int& sequenceLength
) {
    auto& model = g_BuiltModels[modelId];

    auto inputSpan  = bufferToConstantSpan(inputBuffer);
    auto outputSpan = bufferToSpan(outputBuffer);

    g_ModelInputCache[modelId] = cuda_Tensor(inputSpan);

    cuda_Tensor outputTensor(outputSpan.size());
    outputTensor.zeroAll();

    cuda_ComponentsChain::ChainInference(
        *model,
        cuda_ConstantTensorSpan(g_ModelInputCache[modelId]),
        cuda_TensorSpan(outputTensor),
        sequenceLength
    );

    cuda_Tensor::toCPU(outputTensor, outputSpan);
}
void ClearModelInference(
    const std::string& modelId
) {
    auto& model = g_BuiltModels[modelId];

    cuda_ComponentsChain::ClearInference(*model);
}

// Activate
void ActivateModel(
    const std::string& modelId,
    pybind11::buffer inputBuffer,
    pybind11::buffer outputBuffer,
    const int& batchSize,
    pybind11::buffer seriesLengths,
    const int& totalSamples
) {
    auto& model = g_BuiltModels[modelId];

    // Convert buffers → spans
    auto inputSpan = bufferToConstantSpan(inputBuffer);
    auto outputSpan = bufferToSpan(outputBuffer);
    auto lengthsSpan = bufferToConstantLengths(seriesLengths);

    // Copy input → GPU
    // Cache input
   g_ModelInputCache[modelId] = cuda_Tensor(inputSpan);

    // Prepare output tensor
    cuda_Tensor outputTensor(outputSpan.size());
    outputTensor.zeroAll();

    // Activate
    cuda_ComponentsChain::ChainActivation(
        *model,
        cuda_ConstantTensorSpan(g_ModelInputCache[modelId]),
        cuda_TensorSpan(outputTensor),
        batchSize,
        lengthsSpan,
        totalSamples
    );

    // Copy back to CPU
    cuda_Tensor::toCPU(outputTensor, outputSpan);
}

// Backprop
void AdjustModel(
    const std::string& modelId,
    pybind11::buffer outputGradBuffer,
    pybind11::buffer inputGradBuffer,
    const int& batchSize,
    pybind11::buffer seriesLengths,
    const int& totalSamples
) {
    auto& model = g_BuiltModels[modelId];

    // Convert buffers → spans
    auto outputGradSpan = bufferToConstantSpan(outputGradBuffer);
    auto inputGradSpan = bufferToSpan(inputGradBuffer);
    auto lengthsSpan = bufferToConstantLengths(seriesLengths);

    // Copy output gradient → GPU
    cuda_Tensor outputGradTensor(outputGradSpan);

    // Prepare input gradient
    cuda_Tensor inputGradTensor(inputGradSpan.size());
    inputGradTensor.zeroAll();

    // Backprop
    cuda_ComponentsChain::ChainAdjustment(
        *model,
        cuda_ConstantTensorSpan(outputGradTensor),
        cuda_TensorSpan(inputGradTensor),
        batchSize,
        lengthsSpan,
        totalSamples
    );

    // Copy back
    cuda_Tensor::toCPU(inputGradTensor, inputGradSpan);
}

// Update
void UpdateModel(
    const std::string& modelId,
    const cuda_neurologicalValue& learningAlpha
) {
    auto& model = g_BuiltModels[modelId];
    cuda_ComponentsChain::ChainUpdate(*model, learningAlpha);
}

// LOSS FUNCTIONS

void MSE(pybind11::buffer prediction, pybind11::buffer correction, pybind11::buffer propagation) {
    const neurologicalConstantSpan predictionSpan = bufferToConstantSpan(prediction);
    const neurologicalConstantSpan correctionSpan = bufferToConstantSpan(correction);
    const neurologicalSpan propagationSpan = bufferToSpan(propagation);
    MSE_loss_propagation(predictionSpan, correctionSpan, propagationSpan);
}

// EXTRA

void softmax(pybind11::buffer values, const int& eachSize) {
    const neurologicalSpan valuesSpan = bufferToSpan(values);

    int batchSize = valuesSpan.size()/eachSize;

    for (int b = 0; b < batchSize; ++b) {
        cuda_neurologicalValue expSum = 0.0;

        cuda_neurologicalValue maxVal = -1e-38;

        const int start = b*eachSize;
        
        for (int i = start; i < start+eachSize; ++i) {
            maxVal = std::max(maxVal, valuesSpan[i]);
        }

        for (int i = start; i < start+eachSize; ++i) {
            expSum += exp(valuesSpan[i]-maxVal);
        }

        for (int i = start; i < start+eachSize; ++i) {
            valuesSpan[i] = exp(valuesSpan[i]-maxVal)/expSum;
        }
    }
}