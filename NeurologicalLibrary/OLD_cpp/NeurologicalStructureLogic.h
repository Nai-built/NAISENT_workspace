#pragma once

#include <iostream>
#include <vector>
#include <optional>
#include <unordered_map>
#include "AI/Neurological.h"
#include "AI/DenseLayer.h"

using namespace std;

struct AdamParameterOptimizationInfo {
	neurologicalValue v;
    neurologicalValue m;
	int t;
};

using NeurologicalParameterOptimizationSaveInfo = variant
< AdamParameterOptimizationInfo>;

struct NeurologicalParameterSaveInfo {
	neurologicalValue value;
	optional<NeurologicalParameterOptimizationSaveInfo> optimizationInfo;
};

struct DenseNeuronSaveInfo {
    vector<NeurologicalParameterSaveInfo> weights;
    NeurologicalParameterSaveInfo bias;
};

typedef vector<DenseNeuronSaveInfo> DenseNeuronSaveInfoList;

struct DenseLayerSaveInfo;
struct DenseLayerSavChainSaveInfoeInfo;
using NeurologicalComponentSaveInfo = variant
< shared_ptr<DenseLayerSaveInfo>
, shared_ptr<ChainSaveInfo>>;

struct DenseLayerSaveInfo {
    DenseNeuronSaveInfoList denseNeurons;
};
struct ChainSaveInfo {
    unordered_map<int, NeurologicalComponentSaveInfo> chain;
};

struct DenseLayerBuildInfo;
struct ChainBuildInfo;
struct AF_ReLUBuildInfo;
struct AF_TanhBuildInfo;
struct AF_SoftmaxBuildInfo;
using NeurologicalComponentBuildInfo = variant
< shared_ptr<DenseLayerBuildInfo>
, shared_ptr<ChainBuildInfo>
, shared_ptr<AF_ReLUBuildInfo>
, shared_ptr<AF_TanhBuildInfo>
, shared_ptr<AF_SoftmaxBuildInfo>>;

struct DenseLayerBuildInfo {
    int inputSize, outputSize;
};
struct ChainBuildInfo {
    vector<shared_ptr<NeurologicalComponentBuildInfo>> chain;
};
struct AF_ReLUBuildInfo {
    float _negativeLeakage;
};
struct AF_TanhBuildInfo {
    // empty
};
struct AF_SoftmaxBuildInfo {
    bool _ignoreJacobian;
};

struct AdamOptimizerBuildInfo {
	float epsilon;
	float beta1;
	float beta2;
};

using OptimizerBuildInfo = variant
< AdamOptimizerBuildInfo>;

struct NormClippingMethodBuildInfo {
    double clip_value;
};

using ClippingMethodBuildinfo = variant
< NormClippingMethodBuildInfo>;

struct NeurologicalStructureInitializationInfo {
    optional<OptimizerBuildInfo> optimizerBuild;
    optional<ClippingMethodBuildinfo> clippingMethodBuild;
};

NeurologicalComponentSaveInfo ReadNeurologicalStructure(const string& structureId);

string CreateNeurologicalStructure(const NeurologicalComponentBuildInfo& _buildInfo);

void InitializeNeurologicalStructure(const string& structureId, const NeurologicalStructureInitializationInfo& intializationInfo);

NeurologicalPassingValues ActivateNeurologicalStructure(const string& structureId, const NeurologicalPassingValues& input, const string& sampleIndex);
NeurologicalPassingValues AdjustNeurologicalStructure(const string& structureId, const NeurologicalPassingValues& gradient, const string& sampleIndex);
void ApplyNeurologicalStructureAdjustments(const string& structureId, const derivativesApplyingInfo& applyingInfo);