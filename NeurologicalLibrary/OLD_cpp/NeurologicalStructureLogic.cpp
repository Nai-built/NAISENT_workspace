#include "NeurologicalStructureLogic.h"
#include "AI/NeurologicalComponent.h"
#include "AI/Optimization.h"

#include "AI/LossFunctions.h"
#include "AI/NeurologicalComponentsChain.h"
#include "AI/DenseLayer.h"

#include "AI/ActivationFunctions/ReLU.h"
#include "AI/ActivationFunctions/Tanh.h"
#include "AI/ActivationFunctions/Softmax.h"

#include <unordered_map>
#include <variant>

using namespace std;

static int neurologicalStructureCreationIndex = 0;

static unordered_map<string, INeurologicalComponentPTR> neurologicalStructures;

NeurologicalParameterSaveInfo GetParameterSaveInfo(const NeurologicalParameter& parameter) {
    NeurologicalParameterSaveInfo saveInfo;

    saveInfo.value = parameter.value;

    if (auto _ = get_if<ADAM_parameterOptimizationInfo>(&parameter.optimizationInfo)) {
        auto optimizationInfo = *_;
        saveInfo.optimizationInfo = AdamParameterOptimizationInfo {
            .v = optimizationInfo.v,
            .m = optimizationInfo.m,
            .t = optimizationInfo.t,
        };
    }
    
    return saveInfo;
}

template<typename T>
optional<shared_ptr<T>> CheckNeurologicalComponent(INeurologicalComponentPTR component) {
    shared_ptr<T> _cast = dynamic_pointer_cast<T>(component);

    if (_cast == nullptr) {
        return nullopt;
    }

    return _cast;
}

optional<NeurologicalComponentSaveInfo> GetSaveInfoOfNeurologicalComponent(INeurologicalComponentPTR component) {
    optional<NeurologicalComponentSaveInfo> saveInfo = nullopt;

    if (auto denseLayer = CheckNeurologicalComponent<DenseLayer>(component)) {
        DenseNeuronSaveInfoList denseNeurons;

        for (int i = 0; i < denseLayer.value()->neurons.size(); i++) {
            DenseNeuron& neuron = denseLayer.value()->neurons[i];

            vector<NeurologicalParameterSaveInfo> weightsSaveInfo;
            for (int j = 0; j < neuron.weights.size(); j++) {
                weightsSaveInfo.push_back(GetParameterSaveInfo(neuron.weights[j]));
            }
            NeurologicalParameterSaveInfo biasSaveInfo = GetParameterSaveInfo(neuron.bias);

            denseNeurons.push_back({
                .weights = weightsSaveInfo,
                .bias = biasSaveInfo,
            });
        }

        saveInfo.emplace<shared_ptr<DenseLayerSaveInfo>>
        (make_shared<DenseLayerSaveInfo>(DenseLayerSaveInfo {
            .denseNeurons = denseNeurons,
        }));
    } else if (auto chain = CheckNeurologicalComponent<NeurologicalComponentsChain>(component)) {
        unordered_map<int, NeurologicalComponentSaveInfo> componentsChain;
        for (int i = 0; i < chain.value()->chain.size(); i++) {
            optional<NeurologicalComponentSaveInfo>
            elementSaveInfo = GetSaveInfoOfNeurologicalComponent(chain.value()->chain[i]);
            
            if (elementSaveInfo.has_value()) {
                componentsChain[i] = elementSaveInfo.value();
            }
        }

        saveInfo.emplace<shared_ptr<ChainSaveInfo>>
        (make_shared<ChainSaveInfo>(ChainSaveInfo {
            .chain = componentsChain,
        }));
    }

    return saveInfo;
}

NeurologicalComponentSaveInfo ReadNeurologicalStructure(const string& structureId) {
    INeurologicalComponentPTR structure = neurologicalStructures[structureId];

    return GetSaveInfoOfNeurologicalComponent(structure).value();
}

INeurologicalComponentPTR BuildNeurologicalComponent(const NeurologicalComponentBuildInfo& _buildInfo) {
    INeurologicalComponentPTR component;
    
    if (auto _ = get_if<shared_ptr<DenseLayerBuildInfo>>(&_buildInfo)) {
        auto buildInfo = *_;
        component = make_shared<DenseLayer>(buildInfo.inputSize, buildInfo.outputSize);
    }
    else if (auto _ = get_if<shared_ptr<ChainBuildInfo>>(&_buildInfo)) {
        auto buildInfo = *_;
        vector<INeurologicalComponentPTR> _chain;
        
        for (int i = 0; i < buildInfo.chain.size(); i++) {
            _chain.push_back(BuildNeurologicalComponent(*buildInfo.chain[i]));
        }

        component = make_shared<NeurologicalComponentsChain>(_chain);
    }
    else if (shared_ptr<AF_ReLUBuildInfo> _ = get_if<shared_ptr<AF_ReLUBuildInfo>>(&_buildInfo)) {
        AF_ReLUBuildInfo buildInfo = *_;
        component = make_shared<ReLU>(buildInfo._negativeLeakage);
    }
    else if (auto _ = get_if<shared_ptr<AF_TanhBuildInfo>>(&_buildInfo)) {
        auto buildInfo = *_;
        component = make_shared<Tanh>();
    }
    else if (auto _ = get_if<shared_ptr<AF_SoftmaxBuildInfo>>(&_buildInfo)) {
        auto buildInfo = *_;
        component = make_shared<Softmax>(buildInfo._ignoreJacobian);
    } else {
        cout << "NO NEUROLOGICAL STRUCTURE BUILD AVAILABLE FOR THIS BUILD INFO" << endl;
    }

    return component;
}

optional<OptimizerPTR> TryBuildNeurologicalOptimizer(optional<OptimizerBuildInfo> _buildInfo) {
    if (!_buildInfo.has_value()) {
        return nullopt;
    }

    OptimizerPTR optimizer;

    if (auto _ = get_if<AdamOptimizerBuildInfo>(&_buildInfo.value())) {
        auto buildInfo = *_;
        optimizer = make_shared<ADAM_Optimizer>(
            buildInfo.epsilon,
            buildInfo.beta1,
            buildInfo.beta2);
    } else {
        cout << "NO OPTIMIZER BUILD AVAILABLE FOR THIS BUILD INFO" << endl;
    }

    return optimizer;
}

optional<ClippingMethodPTR> TryBuildClippingMethod(optional<ClippingMethodBuildinfo> _buildInfo) {
    if (!_buildInfo.has_value()) {
        return nullopt;
    }

    ClippingMethodPTR clippingMethod;

    if (auto _ = get_if<NormClippingMethodBuildInfo>(&_buildInfo.value())) {
        auto buildInfo = *_;
        clippingMethod = make_shared<NORM_ClippingMethod>(
            buildInfo.clip_value);
    } else {
        cout << "NO CLIPPING METHOD BUILD AVAILABLE FOR THIS BUILD INFO" << endl;
    }

    return clippingMethod;
}

string CreateNeurologicalStructure(const NeurologicalComponentBuildInfo& buildInfo) {
    int _index = neurologicalStructureCreationIndex;
    neurologicalStructureCreationIndex+=1;

    string id = "neurological-structure->" + to_string(_index);
    neurologicalStructures[id] = BuildNeurologicalComponent(buildInfo);
    return id;
}

void InitializeNeurologicalStructure(const string& structureId, const NeurologicalStructureInitializationInfo& intializationInfo) {
    INeurologicalComponentPTR structure = neurologicalStructures[structureId];
    
    structure->initialize({
        .optimizer = TryBuildNeurologicalOptimizer(intializationInfo.optimizerBuild),
        .clippingMethod = TryBuildClippingMethod(intializationInfo.clippingMethodBuild),
    });
}

NeurologicalPassingValues ActivateNeurologicalStructure(const string& structureId, const NeurologicalPassingValues& input, const string& sampleIndex) {
    INeurologicalComponentPTR structure = neurologicalStructures[structureId];

    NeurologicalPassingValues output = structure->forwardPass(input, sampleIndex);

    return output;
}
NeurologicalPassingValues AdjustNeurologicalStructure(const string& structureId, const NeurologicalPassingValues& gradient, const string& sampleIndex) {
    INeurologicalComponentPTR structure = neurologicalStructures[structureId];

    return structure->backPropagation(gradient, sampleIndex);
}
void ApplyNeurologicalStructureAdjustments(const string& structureId, const derivativesApplyingInfo& applyingInfo) {
    INeurologicalComponentPTR structure = neurologicalStructures[structureId];

    structure->applyDerivatives(applyingInfo);
}