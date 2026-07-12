#pragma once

#include <string>

#include "INeurologicalComponent.h"

#include "DenseLayer.h"
#include "ConvolutionalLayer.h"
#include "LSTM_RecursiveLayer.h"
#include "ComponentsChain.h"
#include "DecoderOnly_Transformer.h"

struct BaseChannelsPoolerBuildInfo
{
    ChannelsPoolerTypes poolerType;

    std::string poolingSize;
    int stride;

    virtual ~BaseChannelsPoolerBuildInfo() = default; // Make the base polymorphic
};

struct MaxPool__BuildInfo : BaseChannelsPoolerBuildInfo {

};
struct AveragePool__BuildInfo : BaseChannelsPoolerBuildInfo {
    
};
struct MinPool__BuildInfo : BaseChannelsPoolerBuildInfo {
    
};

struct BaseActivationFunctionBuildInfo
{
public:
    ActivationFunctionTypes activationType;

    virtual ~BaseActivationFunctionBuildInfo() = default; // Make the base polymorphic
};

struct ReLU__BuildInfo : BaseActivationFunctionBuildInfo {
    neurologicalValue leakage;
};

struct BaseNeurologicalComponentBuildInfo
{
public:
    NeurologicalComponentTypes componentType;

    virtual ~BaseNeurologicalComponentBuildInfo() = default; // Make the base polymorphic
};

struct DenseLayerBuildInfo : BaseNeurologicalComponentBuildInfo {
    int inputSize;
    int outputSize;

    std::shared_ptr<BaseActivationFunctionBuildInfo> activationFunction = nullptr;
};

struct ConvolutionalLayerBuildInfo : BaseNeurologicalComponentBuildInfo {
    int inputChannels;
    int outputChannels;

    int stride;
    int padding;

    std::string kernelSize;
    std::string inputChannelSize;

    std::shared_ptr<BaseActivationFunctionBuildInfo> activationFunction = nullptr;
    std::shared_ptr<BaseChannelsPoolerBuildInfo> channelsPooler = nullptr;
};

struct LSTM_RecursiveLayerBuildInfo : BaseNeurologicalComponentBuildInfo {
    int inputSize;
    int outputSize;

    bool castBeyond;
};

struct ComponentsChainBuildInfo : BaseNeurologicalComponentBuildInfo {
    std::vector<std::shared_ptr<BaseNeurologicalComponentBuildInfo>> componentsChainBuild;
};

// TRANSFORMERS
struct HiddenFFL_BuildInfo {
    int hidden_size;
    std::shared_ptr<BaseActivationFunctionBuildInfo> activationFunction = nullptr;
};
struct TransformerStack_BuildInfo {
    int attention_heads;
    std::vector<std::shared_ptr<HiddenFFL_BuildInfo>> ffn;
};

struct DecoderOnly_TransformerBuildInfo : BaseNeurologicalComponentBuildInfo {
    int inputTokens;
    int outputTokens;
    int encodingSize;

    std::vector<std::shared_ptr<TransformerStack_BuildInfo>> stacksBuild;
};

struct BaseNeurologicalComponentSaveInfo
{
public:
    std::string saveType;

    virtual ~BaseNeurologicalComponentSaveInfo() = default; // Make the base polymorphic
};

typedef std::shared_ptr<BaseNeurologicalComponentSaveInfo> SaveInfoPTR;

struct CommonLayerSaveInfo : BaseNeurologicalComponentSaveInfo {
    std::vector<neurologicalValue> weights;
    std::vector<neurologicalValue> biases;
};
struct TransformerSaveInfo : BaseNeurologicalComponentSaveInfo {
    neurologicalBuffer encodeWeights;
    
    std::vector<neurologicalValue> attentionWeights;
    std::vector<neurologicalValue> attentionFinalBiases;

    std::vector<neurologicalValue> normalizationGammas;

    std::vector<neurologicalValue> feedForwardNetworksWeights;
    std::vector<neurologicalValue> feedForwardNetworksBiases;

    std::vector<neurologicalValue> decodeWeights;
    std::vector<neurologicalValue> decodeBiases;
};

struct ComponentsChainSaveInfo : BaseNeurologicalComponentSaveInfo {
    std::vector<SaveInfoPTR> componentsChainSave;
};

SaveInfoPTR ReadNeurologicalStructure(const std::string& id);
void WriteNeurologicalStructure(const std::string& id, const SaveInfoPTR& saveInfo);

std::string BuildNeurologicalStructure(const BaseNeurologicalComponentBuildInfo& buildInfo);
void InitializeNeurologicalStructure(const std::string& id);
void ActivateNeurologicalStructure(const std::string& id, const neurologicalConstantSpan input, const neurologicalSpan output, const int& batchSize
    , const lengths seriesLengths, const int& totalSamples);
void AdjustNeurologicalStructure(const std::string& id, const neurologicalSpan propagation, const neurologicalSpan inputPropagation, const neurologicalValue& learnRate, const int& batchSize
    , const lengths seriesLengths, const int& totalSamples);

void softmax(const neurologicalSpan values, const int& eachSize);