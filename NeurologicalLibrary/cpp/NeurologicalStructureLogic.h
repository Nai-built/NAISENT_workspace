#pragma once

#include <iostream>
#include <vector>
#include <optional>
#include "AI/Neurological.h"
#include "AI/NeurologicalComponentsChain.h"

#include <variant>

using namespace std;

using NeurologicalStructure = shared_ptr<NeurologicalComponentsChain>;

struct NeurologicalParameterOptimizationSaveInfo {
	string optimizerType;

	// ADAM OPTIMIZER
	optional<neurologicalValue> v;
	optional<neurologicalValue> m;
	optional<int> t;
};

struct NeurologicalParameterSaveInfo {
	neurologicalValue value;
	optional<NeurologicalParameterOptimizationSaveInfo> optimizationInfo;
};

struct DenseNeuronSaveInfo {
    vector<NeurologicalParameterSaveInfo> weights;
    NeurologicalParameterSaveInfo bias;
};

typedef vector<vector<NeurologicalParameterSaveInfo>> KernelSaveInfo;
struct ConvolutionalFilterSaveInfo {
    vector<KernelSaveInfo> kernels;
    NeurologicalParameterSaveInfo bias;
};

typedef vector<DenseNeuronSaveInfo> DenseLayerSaveInfo;
typedef vector<ConvolutionalFilterSaveInfo> ConvolutionalLayerSaveInfo;

struct NeurologicalComponentSaveInfo {
    string componentType;

    // DENSE LAYER INFO
    optional<DenseLayerSaveInfo> denseNeurons;
    
    // LSTM RECURSIVE LAYER INFO
    optional<DenseLayerSaveInfo> forgetGateNeurons;
    optional<DenseLayerSaveInfo> inputGateNeurons;
    optional<DenseLayerSaveInfo> candidateMemoryGateNeurons;
    optional<DenseLayerSaveInfo> outputGateNeurons;

    // CONVOLUTIONAL LAYER INFO
    optional<ConvolutionalLayerSaveInfo> convolutionalFilters;

    // CHAIN INFO
    optional<vector<NeurologicalComponentSaveInfo>> chain;
    
    // -- ACTIVATION FUNCTIONS --

    // empty

    // -- EXTRAS --

    // empty
};

struct NeurologicalComponentBuildInfo {
    string componentType;

    // DENSE LAYER/LSTM_RECURSIVE LAYER/CONVOLUTIONAL LAYER INFO
    optional<int> inputSize, outputSize;

    // CONVOLUTIONAL LAYER AND CONVOLUTIONAL POOL INFO
    optional<int> stride;

    // CONVOLUTIONAL LAYER INFO
    optional<int> padding;
    optional<string> kernelSize;

    // CONVOLUTIONAL POOL INFO
    optional<string> poolSize;
    optional<string> poolingType;

    // CHANNELS FLATTENER -empty

    // CHANNELS SPREADER
    optional<int> expectedChannelsAmount, expectedChannelsWidth, expectedChannelsHeight;
    
    // CHAIN INFO
    optional<vector<NeurologicalComponentBuildInfo>> chain;

    // -- ACTIVATION FUNCTIONS --
    // ReLU
    optional<float> _negativeLeakage;

    // Tanh -empty

    // Softmax
    optional<bool> _ignoreJacobian;
};

struct OptimizerBuildInfo {
    string optimizerType;

    // ADAM
	optional<float> epsilon;
	optional<float> beta1;
	optional<float> beta2;
};

struct ClippingMethodBuildinfo {
    string clippingMethodType;

    // NORM
    optional<neurologicalValue> clip_value;
};

struct NeurologicalStructureInitializationInfo {
    optional<OptimizerBuildInfo> optimizerBuild;
    optional<ClippingMethodBuildinfo> clippingMethodBuild;
};

NeurologicalComponentSaveInfo ReadNeurologicalStructure(const string& structureId);
void WriteNeurologicalStructure(const string& structureId, const NeurologicalComponentSaveInfo& saveInfo);

string CreateNeurologicalStructure(const NeurologicalComponentBuildInfo& buildInfo);

void InitializeNeurologicalStructure(const string& structureId, const NeurologicalStructureInitializationInfo& intializationInfo);

NeurologicalPassingValues ActivateNeurologicalStructure(const string& structureId, const NeurologicalPassingValues& input, const string& sampleIndex);
NeurologicalPassingValues AdjustNeurologicalStructure(const string& structureId, const NeurologicalPassingValues& gradient, const string& sampleIndex);
void ApplyNeurologicalStructureAdjustments(const string& structureId, const derivativesApplyingInfo& applyingInfo);