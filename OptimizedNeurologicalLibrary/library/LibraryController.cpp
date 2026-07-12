#include <unordered_map>
#include <iostream>

#include "ActivationFunctions/ReLU.h"

#include "ChannelsPoolers/Max.h"
#include "ChannelsPoolers/Average.h"
#include "ChannelsPoolers/Min.h"

#include "LibraryController.h"

constexpr neurologicalValue LOWEST = -3.40282e+38;

std::unordered_map<std::string, NeurologicalComponent_UNIQUE> structures =
std::unordered_map<std::string, NeurologicalComponent_UNIQUE>();
int neurologicalStructureCreationIndex = 0;

void GetNeurologicalStructureSaveInfo(INeurologicalComponent* _component, SaveInfoPTR& _saveInfo) {
    switch(_component->componentType) {
        case NeurologicalComponentTypes::DECODER_ONLY_TRANSFORMER: {
            DecoderOnly_Transformer* transformer =
            static_cast<DecoderOnly_Transformer*>(_component);

            _saveInfo = make_shared<TransformerSaveInfo>();

            std::shared_ptr<TransformerSaveInfo> transformerSaveInfo =
            std::static_pointer_cast<TransformerSaveInfo>(_saveInfo);
            transformerSaveInfo->saveType = "transformer";

            transformerSaveInfo->encodeWeights = transformer->encodeWeights;

            transformerSaveInfo->attentionWeights = transformer->attentionWeights;
            transformerSaveInfo->attentionFinalBiases = transformer->attentionFinalBiases;

            transformerSaveInfo->normalizationGammas = transformer->normalizationGammas;

            transformerSaveInfo->feedForwardNetworksWeights = transformer->feedForwardNetworksWeights;
            transformerSaveInfo->feedForwardNetworksBiases = transformer->feedForwardNetworksBiases;
            
            transformerSaveInfo->decodeWeights = transformer->decodeWeights;
            transformerSaveInfo->decodeBiases = transformer->decodeBiases;
            
            // std::cout << "save dense weight: " << layer->weights[0] << std::endl;
            break;
        }
        case NeurologicalComponentTypes::COMPONENTS_CHAIN: {
            ComponentsChain* chain =
            static_cast<ComponentsChain*>(_component);

            _saveInfo = make_shared<ComponentsChainSaveInfo>();

            std::shared_ptr<ComponentsChainSaveInfo> chainSaveInfo =
            std::static_pointer_cast<ComponentsChainSaveInfo>(_saveInfo);
            chainSaveInfo->saveType = "chain";
            chainSaveInfo->componentsChainSave = std::vector<SaveInfoPTR>(chain->componentsChain.size());
            
            for (int i = 0; i < chain->componentsChain.size(); ++i) {
                std::cout << "save: " << i << std::endl;
                GetNeurologicalStructureSaveInfo
                    (chain->componentsChain[i].get(), chainSaveInfo->componentsChainSave[i]);
            }
            break;
        }
        case NeurologicalComponentTypes::DENSE_LAYER: {
            DenseLayer* layer =
            static_cast<DenseLayer*>(_component);

            _saveInfo = make_shared<CommonLayerSaveInfo>();

            std::shared_ptr<CommonLayerSaveInfo> commonLayerSaveInfo =
            std::static_pointer_cast<CommonLayerSaveInfo>(_saveInfo);
            commonLayerSaveInfo->saveType = "common-layer";

            commonLayerSaveInfo->weights = layer->weights;
            commonLayerSaveInfo->biases = layer->biases;
            
            // std::cout << "save dense weight: " << layer->weights[0] << std::endl;
            break;
        }
        case NeurologicalComponentTypes::CONVOLUTIONAL_LAYER: {
            ConvolutionalLayer* layer =
            static_cast<ConvolutionalLayer*>(_component);

            _saveInfo = make_shared<CommonLayerSaveInfo>();

            std::shared_ptr<CommonLayerSaveInfo> commonLayerSaveInfo =
            std::static_pointer_cast<CommonLayerSaveInfo>(_saveInfo);
            commonLayerSaveInfo->saveType = "common-layer";

            commonLayerSaveInfo->weights = layer->weights;
            commonLayerSaveInfo->biases = layer->biases;

            // std::cout << "save convolutional weight: " << layer->weights[0] << std::endl;
            break;
        }
        case NeurologicalComponentTypes::RECURSIVE_LSTM: {
            LSTM_RecursiveLayer* layer =
            static_cast<LSTM_RecursiveLayer*>(_component);

            _saveInfo = make_shared<CommonLayerSaveInfo>();

            std::shared_ptr<CommonLayerSaveInfo> commonLayerSaveInfo =
            std::static_pointer_cast<CommonLayerSaveInfo>(_saveInfo);
            commonLayerSaveInfo->saveType = "common-layer";

            commonLayerSaveInfo->weights = layer->weights;
            commonLayerSaveInfo->biases = layer->biases;
            
            // std::cout << "save lstm weight: " << layer->weights[0] << std::endl;
            break;
        }
    }
}
void SetNeurologicalStructureSaveInfo(INeurologicalComponent* _component, SaveInfoPTR _saveInfo) {
    switch(_component->componentType) {
        case NeurologicalComponentTypes::DECODER_ONLY_TRANSFORMER: {
            DecoderOnly_Transformer* transformer =
            static_cast<DecoderOnly_Transformer*>(_component);

            std::shared_ptr<TransformerSaveInfo> transformerSaveInfo =
            std::static_pointer_cast<TransformerSaveInfo>(_saveInfo);

            transformer->encodeWeights = transformerSaveInfo->encodeWeights;

            transformer->attentionWeights = transformerSaveInfo->attentionWeights;
            transformer->attentionFinalBiases = transformerSaveInfo->attentionFinalBiases;

            transformer->normalizationGammas = transformerSaveInfo->normalizationGammas;

            transformer->feedForwardNetworksWeights = transformerSaveInfo->feedForwardNetworksWeights;
            transformer->feedForwardNetworksBiases = transformerSaveInfo->feedForwardNetworksBiases;
            
            transformer->decodeWeights = transformerSaveInfo->decodeWeights;
            transformer->decodeBiases = transformerSaveInfo->decodeBiases;
            
            // std::cout << "load dense weight: " << layer->weights[0] << std::endl;
            break;
        }
        case NeurologicalComponentTypes::COMPONENTS_CHAIN: {
            ComponentsChain* chain =
            static_cast<ComponentsChain*>(_component);

            std::shared_ptr<ComponentsChainSaveInfo> chainSaveInfo =
            std::static_pointer_cast<ComponentsChainSaveInfo>(_saveInfo);
            
            for (int i = 0; i < chain->componentsChain.size(); ++i) {
                std::cout << "can load?: " << i << std::endl;
                if (i >= chainSaveInfo->componentsChainSave.size()) break;
                std::cout << "load: " << i << std::endl;
                SetNeurologicalStructureSaveInfo
                    (chain->componentsChain[i].get(), chainSaveInfo->componentsChainSave[i]);
            }
            break;
        }
        case NeurologicalComponentTypes::DENSE_LAYER: {
            DenseLayer* layer =
            static_cast<DenseLayer*>(_component);

            std::shared_ptr<CommonLayerSaveInfo> commonLayerSaveInfo =
            std::static_pointer_cast<CommonLayerSaveInfo>(_saveInfo);

            layer->weights = commonLayerSaveInfo->weights;
            layer->biases = commonLayerSaveInfo->biases;
            
            // std::cout << "load dense weight: " << layer->weights[0] << std::endl;
            break;
        }
        case NeurologicalComponentTypes::CONVOLUTIONAL_LAYER: {
            ConvolutionalLayer* layer =
            static_cast<ConvolutionalLayer*>(_component);

            std::shared_ptr<CommonLayerSaveInfo> commonLayerSaveInfo =
            std::static_pointer_cast<CommonLayerSaveInfo>(_saveInfo);

            layer->weights = commonLayerSaveInfo->weights;
            layer->biases = commonLayerSaveInfo->biases;
            
            // std::cout << "load convolutional weight: " << layer->weights[0] << std::endl;
            break;
        }
        case NeurologicalComponentTypes::RECURSIVE_LSTM: {
            LSTM_RecursiveLayer* layer =
            static_cast<LSTM_RecursiveLayer*>(_component);

            std::shared_ptr<CommonLayerSaveInfo> commonLayerSaveInfo =
            std::static_pointer_cast<CommonLayerSaveInfo>(_saveInfo);

            layer->weights = commonLayerSaveInfo->weights;
            layer->biases = commonLayerSaveInfo->biases;
            
            // std::cout << "load lstm weight: " << layer->weights[0] << std::endl;
            break;
        }
    }
}

SaveInfoPTR ReadNeurologicalStructure(const std::string& id) {
    SaveInfoPTR saveInfo;

    GetNeurologicalStructureSaveInfo(structures[id].get(), saveInfo);

    return saveInfo;
}
void WriteNeurologicalStructure(const std::string& id, const SaveInfoPTR& saveInfo) {
    SetNeurologicalStructureSaveInfo(structures[id].get(), saveInfo);
}

std::unique_ptr<IActivationFunction> buildActivationFunction(const BaseActivationFunctionBuildInfo& buildInfo) {
    switch(buildInfo.activationType) {
        case ActivationFunctionTypes::RELU: {
            const ReLU__BuildInfo& reluBuildInfo =
            static_cast<const ReLU__BuildInfo&>(buildInfo);

            return std::make_unique<ReLU>(reluBuildInfo.leakage);
            break;
        }
    }
}
std::unique_ptr<IChannelsPooler> buildChannelsPooler(const BaseChannelsPoolerBuildInfo& buildInfo) {
    switch(buildInfo.poolerType) {
        case ChannelsPoolerTypes::MAX: {
            const MaxPool__BuildInfo& maxBuildInfo =
            static_cast<const MaxPool__BuildInfo&>(buildInfo);

            vector<std::string> splitedPoolingSizeString = splitStringByString(maxBuildInfo.poolingSize, "x");

            return std::make_unique<Max>(stoi(splitedPoolingSizeString[0]), stoi(splitedPoolingSizeString[1])
            , maxBuildInfo.stride);
            break;
        }
        case ChannelsPoolerTypes::AVERAGE: {
            const AveragePool__BuildInfo& averageBuildInfo =
            static_cast<const AveragePool__BuildInfo&>(buildInfo);

            vector<std::string> splitedPoolingSizeString = splitStringByString(averageBuildInfo.poolingSize, "x");

            return std::make_unique<Average>(stoi(splitedPoolingSizeString[0]), stoi(splitedPoolingSizeString[1])
            , averageBuildInfo.stride);
            break;
        }
        case ChannelsPoolerTypes::MIN: {
            const MinPool__BuildInfo& minBuildInfo =
            static_cast<const MinPool__BuildInfo&>(buildInfo);

            vector<std::string> splitedPoolingSizeString = splitStringByString(minBuildInfo.poolingSize, "x");

            return std::make_unique<Min>(stoi(splitedPoolingSizeString[0]), stoi(splitedPoolingSizeString[1])
            , minBuildInfo.stride);
            break;
        }
    }
}

TransformerStack buildTransformerStack(const TransformerStack_BuildInfo& buildInfo) {
    std::vector<int> hidden_ffn = std::vector<int>(buildInfo.ffn.size());
    std::vector<ActivationFunction_UNIQUE> activations_ffn = std::vector<ActivationFunction_UNIQUE>(buildInfo.ffn.size());

    for (int i = 0; i < buildInfo.ffn.size(); i++) {
        hidden_ffn[i] = buildInfo.ffn[i]->hidden_size;
        ActivationFunction_UNIQUE activation = nullptr;
        if (buildInfo.ffn[i]->activationFunction != nullptr && buildInfo.ffn[i]->activationFunction->activationType != ActivationFunctionTypes::LINEAR) {
            activation = std::move(buildActivationFunction(*buildInfo.ffn[i]->activationFunction.get()));
        } else {
            activation = std::make_unique<IActivationFunction>();
            activation->activationType = ActivationFunctionTypes::LINEAR;
        }
        activations_ffn[i] = std::move(activation);
    }

    return {.attentionHeads = buildInfo.attention_heads, .hidden_FFN = hidden_ffn, .activations_FFN = std::move(activations_ffn)};
}

void BuildNeurologicalComponent(const BaseNeurologicalComponentBuildInfo& buildInfo, NeurologicalComponent_UNIQUE& _component) {
    switch(buildInfo.componentType) {
        case NeurologicalComponentTypes::DECODER_ONLY_TRANSFORMER: {
            const DecoderOnly_TransformerBuildInfo& transformerBuildInfo =
            static_cast<const DecoderOnly_TransformerBuildInfo&>(buildInfo);

            _component = std::make_unique<DecoderOnly_Transformer>(transformerBuildInfo.inputTokens, transformerBuildInfo.outputTokens, transformerBuildInfo.encodingSize);
            // std::cout << "size: " << chainBuildInfo.componentsChainBuild.size() << std::endl;
            // std::cout << "t: " << static_cast<int>(chainBuildInfo.componentsChainBuild[0].componentType) << std::endl;
            DecoderOnly_Transformer* transformer =
            static_cast<DecoderOnly_Transformer*>(_component.get());
            transformer->stacks = std::vector<TransformerStack>(transformerBuildInfo.stacksBuild.size());
            for (int i = 0; i < transformerBuildInfo.stacksBuild.size(); ++i) {
                transformer->stacks[i] = std::move(buildTransformerStack(*transformerBuildInfo.stacksBuild[i]));
            }
            break;
        }
        case NeurologicalComponentTypes::COMPONENTS_CHAIN: {
            const ComponentsChainBuildInfo& chainBuildInfo =
            static_cast<const ComponentsChainBuildInfo&>(buildInfo);

            _component = std::make_unique<ComponentsChain>();
            // std::cout << "size: " << chainBuildInfo.componentsChainBuild.size() << std::endl;
            // std::cout << "t: " << static_cast<int>(chainBuildInfo.componentsChainBuild[0].componentType) << std::endl;
            ComponentsChain* chain =
            static_cast<ComponentsChain*>(_component.get());
            chain->componentsChain = std::vector<NeurologicalComponent_UNIQUE>(chainBuildInfo.componentsChainBuild.size());
            for (int i = 0; i < chainBuildInfo.componentsChainBuild.size(); ++i) {
                BuildNeurologicalComponent
                    (*chainBuildInfo.componentsChainBuild[i].get(), chain->componentsChain[i]);
            }
            break;
        }
        case NeurologicalComponentTypes::DENSE_LAYER: {
            const DenseLayerBuildInfo& denseLayerBuildInfo =
            static_cast<const DenseLayerBuildInfo&>(buildInfo);

            if (denseLayerBuildInfo.activationFunction != nullptr && denseLayerBuildInfo.activationFunction->activationType != ActivationFunctionTypes::LINEAR) {
                _component = std::make_unique<DenseLayer>
                (denseLayerBuildInfo.inputSize, denseLayerBuildInfo.outputSize
                , buildActivationFunction(*denseLayerBuildInfo.activationFunction.get()));
            } else {
                _component = std::make_unique<DenseLayer>(denseLayerBuildInfo.inputSize, denseLayerBuildInfo.outputSize, nullptr);
            }
            break;
        }
        case NeurologicalComponentTypes::CONVOLUTIONAL_LAYER: {
            const ConvolutionalLayerBuildInfo& convolutionalLayerBuildInfo =
            static_cast<const ConvolutionalLayerBuildInfo&>(buildInfo);
            
            vector<std::string> splitedKernelSizeString = splitStringByString(convolutionalLayerBuildInfo.kernelSize, "x");
            vector<std::string> splitedInputChannelSizeString = splitStringByString(convolutionalLayerBuildInfo.inputChannelSize, "x");

            ActivationFunction_UNIQUE activation = nullptr;
            if (convolutionalLayerBuildInfo.activationFunction != nullptr && convolutionalLayerBuildInfo.activationFunction->activationType != ActivationFunctionTypes::LINEAR) {
                activation = std::move(buildActivationFunction(*convolutionalLayerBuildInfo.activationFunction.get()));
            }
            ChannelsPooler_UNIQUE pooler = nullptr;
            if (convolutionalLayerBuildInfo.channelsPooler != nullptr && convolutionalLayerBuildInfo.channelsPooler->poolerType != ChannelsPoolerTypes::NONE) {
                pooler = std::move(buildChannelsPooler(*convolutionalLayerBuildInfo.channelsPooler.get()));
            }

            _component = make_unique<ConvolutionalLayer>(
                convolutionalLayerBuildInfo.inputChannels, convolutionalLayerBuildInfo.outputChannels
                , stoi(splitedKernelSizeString[0]), stoi(splitedKernelSizeString[1])
                , stoi(splitedInputChannelSizeString[0]), stoi(splitedInputChannelSizeString[1])
                , convolutionalLayerBuildInfo.stride, convolutionalLayerBuildInfo.padding
                , std::move(activation)
                , std::move(pooler));
            break;
        }
        case NeurologicalComponentTypes::RECURSIVE_LSTM: {
            const LSTM_RecursiveLayerBuildInfo& lstm_recursiveLayerBuildInfo =
            static_cast<const LSTM_RecursiveLayerBuildInfo&>(buildInfo);

            _component = std::make_unique<LSTM_RecursiveLayer>(lstm_recursiveLayerBuildInfo.inputSize
                , lstm_recursiveLayerBuildInfo.outputSize, lstm_recursiveLayerBuildInfo.castBeyond);
            break;
        }
    }
}

void InitializeNeurologicalComponent(INeurologicalComponent* _component) {
    switch(_component->componentType) {
        case NeurologicalComponentTypes::DECODER_ONLY_TRANSFORMER: {
            DecoderOnly_Transformer* transformer =
            static_cast<DecoderOnly_Transformer*>(_component);

            DecoderOnly_Transformer::Initialize(transformer);
        }
        case NeurologicalComponentTypes::COMPONENTS_CHAIN: {
            ComponentsChain* chain =
            static_cast<ComponentsChain*>(_component);

            for (int i = 0; i < chain->componentsChain.size(); ++i) {
                InitializeNeurologicalComponent(chain->componentsChain[i].get());
            }
            ComponentsChain::Initialize(chain);
            break;
        }
        break;
        case NeurologicalComponentTypes::DENSE_LAYER: {
            DenseLayer* denseLayer =
            static_cast<DenseLayer*>(_component);

            DenseLayer::Initialize(denseLayer);
            break;
        }
        case NeurologicalComponentTypes::CONVOLUTIONAL_LAYER: {
            ConvolutionalLayer* convolutionalLayer =
            static_cast<ConvolutionalLayer*>(_component);

            ConvolutionalLayer::Initialize(convolutionalLayer);
            break;
        }
        case NeurologicalComponentTypes::RECURSIVE_LSTM: {
            LSTM_RecursiveLayer* lstm_recursiveLayer =
            static_cast<LSTM_RecursiveLayer*>(_component);

            LSTM_RecursiveLayer::Initialize(lstm_recursiveLayer);
            break;
        }
    }
}

void ActivateNeurologicalComponent(INeurologicalComponent* _component, const neurologicalConstantSpan input, const neurologicalSpan output, const int& batchSize
    , const lengths seriesLengths, const int& totalSamples)
{
    switch(_component->componentType) {
        case NeurologicalComponentTypes::DECODER_ONLY_TRANSFORMER: {
            DecoderOnly_Transformer* transformer =
            static_cast<DecoderOnly_Transformer*>(_component);

            DecoderOnly_Transformer::ForwardPass(transformer, input, output, batchSize, seriesLengths, totalSamples);
            break;
        }
        case NeurologicalComponentTypes::COMPONENTS_CHAIN: {
            ComponentsChain* chain =
            static_cast<ComponentsChain*>(_component);

            ComponentsChain::ActivateChain(chain, input, output, batchSize, seriesLengths, totalSamples);
            break;
        }
        break;
        case NeurologicalComponentTypes::DENSE_LAYER: {
            DenseLayer* denseLayer =
            static_cast<DenseLayer*>(_component);

            DenseLayer::ForwardPass(denseLayer, input, output, batchSize);
            break;
        }
        case NeurologicalComponentTypes::CONVOLUTIONAL_LAYER: {
            ConvolutionalLayer* convolutionalLayer =
            static_cast<ConvolutionalLayer*>(_component);
            
            ConvolutionalLayer::ForwardPass(convolutionalLayer, input, output, batchSize);
        }
        case NeurologicalComponentTypes::RECURSIVE_LSTM: {
            LSTM_RecursiveLayer* lstm_recursiveLayer =
            static_cast<LSTM_RecursiveLayer*>(_component);
            
            LSTM_RecursiveLayer::ForwardPass(lstm_recursiveLayer, input, output, batchSize, seriesLengths, totalSamples);
        }
    }
}

void AdjustNeurologicalComponent(INeurologicalComponent* _component, const neurologicalSpan propagation, const neurologicalSpan inputPropagation, const neurologicalValue& learnRate, const int& batchSize
    , const lengths seriesLengths, const int& totalSamples)
{
    switch(_component->componentType) {
        case NeurologicalComponentTypes::DECODER_ONLY_TRANSFORMER: {
            DecoderOnly_Transformer* transformer =
            static_cast<DecoderOnly_Transformer*>(_component);

            DecoderOnly_Transformer::BackPropagation(transformer, propagation, inputPropagation, learnRate, batchSize, seriesLengths, totalSamples);
            break;
        }
        case NeurologicalComponentTypes::COMPONENTS_CHAIN: {
            ComponentsChain* chain =
            static_cast<ComponentsChain*>(_component);

            ComponentsChain::AdjustChain(chain, propagation, inputPropagation, learnRate, batchSize, seriesLengths, totalSamples);
            break;
        }
        break;
        case NeurologicalComponentTypes::DENSE_LAYER: {
            DenseLayer* denseLayer =
            static_cast<DenseLayer*>(_component);

            DenseLayer::BackPropagation(denseLayer, propagation, inputPropagation, learnRate, batchSize);
            break;
        }
        case NeurologicalComponentTypes::CONVOLUTIONAL_LAYER: {
            ConvolutionalLayer* convolutionalLayer =
            static_cast<ConvolutionalLayer*>(_component);
            
            ConvolutionalLayer::BackPropagation(convolutionalLayer, propagation, inputPropagation, learnRate, batchSize);
        }
        case NeurologicalComponentTypes::RECURSIVE_LSTM: {
            LSTM_RecursiveLayer* lstm_recursiveLayer =
            static_cast<LSTM_RecursiveLayer*>(_component);
            
            LSTM_RecursiveLayer::BackPropagation(lstm_recursiveLayer, propagation, inputPropagation, learnRate, batchSize, seriesLengths, totalSamples);
        }
    }
}

std::string BuildNeurologicalStructure(const BaseNeurologicalComponentBuildInfo& buildInfo) {
    int _index = neurologicalStructureCreationIndex;
    neurologicalStructureCreationIndex+=1;

    std::string id = "neurological-structure->" + std::to_string(_index);
    BuildNeurologicalComponent(buildInfo, structures[id]);

    return id;
}

void InitializeNeurologicalStructure(const std::string& id) {
    InitializeNeurologicalComponent(structures[id].get());
}

void ActivateNeurologicalStructure(const std::string& id, const neurologicalConstantSpan input, const neurologicalSpan output, const int& batchSize
    , const lengths seriesLengths, const int& totalSamples)
{
    ActivateNeurologicalComponent(structures[id].get(), input, output, batchSize, seriesLengths, totalSamples);
}
void AdjustNeurologicalStructure(const std::string& id, const neurologicalSpan propagation, const neurologicalSpan inputPropagation, const neurologicalValue& learnRate, const int& batchSize
    , const lengths seriesLengths, const int& totalSamples)
{
    AdjustNeurologicalComponent(structures[id].get(), propagation, inputPropagation, learnRate, batchSize, seriesLengths, totalSamples);
}

void softmax(const neurologicalSpan values, const int& eachSize) {
    int batchSize = values.size()/eachSize;

    for (int b = 0; b < batchSize; ++b) {
        neurologicalValue expSum = 0.0;

        neurologicalValue maxVal = LOWEST;

        const int start = b*eachSize;
        
        for (int i = start; i < start+eachSize; ++i) {
            maxVal = std::max(maxVal, values[i]);
        }

        for (int i = start; i < start+eachSize; ++i) {
            expSum += exp(values[i]-maxVal);
        }

        for (int i = start; i < start+eachSize; ++i) {
            values[i] = exp(values[i]-maxVal)/expSum;
        }
    }
}