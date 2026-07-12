#include "NeurologicalStructureLogic.h"
#include "AI/NeurologicalComponent.h"
#include "AI/Optimization.h"

#include "AI/LossFunctions.h"

#include "AI/NeurologicalComponentsChain.h"

#include "AI/DenseLayer.h"
#include "AI/LSTM_RecursiveLayer.h"
#include "AI/ConvolutionalLayer.h"

#include "AI/ConvolutionalPool.h"
#include "AI/ChannelsFlattener.h"
#include "AI/ChannelsSpreader.h"

#include "AI/ActivationFunctions/ReLU.h"
#include "AI/ActivationFunctions/Tanh.h"
#include "AI/ActivationFunctions/Softmax.h"

#include <unordered_map>

using namespace std;

static int neurologicalStructureCreationIndex = 0;

static unordered_map<string, NeurologicalStructure> neurologicalStructures;

NeurologicalParameterSaveInfo GetParameterSaveInfo(const NeurologicalParameter& parameter) {
    NeurologicalParameterSaveInfo saveInfo;

    saveInfo.value = parameter.value;

    if (auto adamOptimizationInfo = get_if<ADAM_parameterOptimizationInfo>(&parameter.optimizationInfo)) {
        saveInfo.optimizationInfo = {
            .optimizerType = "ADAM",
            .v = (*adamOptimizationInfo).v,
            .m = (*adamOptimizationInfo).m,
            .t = (*adamOptimizationInfo).t,
        };
    } else {
        saveInfo.optimizationInfo = nullopt;
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

NeurologicalComponentSaveInfo GetSaveInfoOfNeurologicalComponent(INeurologicalComponentPTR component) {
    NeurologicalComponentSaveInfo saveInfo;

    if (auto denseLayer = CheckNeurologicalComponent<DenseLayer>(component)) {
        // save data from dense layer
        DenseLayerSaveInfo denseNeurons;

        for (int i = 0; i < denseLayer.value()->neurons.size(); i++) {
            const DenseNeuron& neuron = denseLayer.value()->neurons[i];

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

        saveInfo = {
            .componentType = "DENSE-LAYER",
            .denseNeurons = denseNeurons,
        };
    } else if (auto LSTM_recursiveLayer = CheckNeurologicalComponent<LSTM_RecursiveLayer>(component)) {
        // save data from lstm recursive layer
        saveInfo = {
            .componentType = "RECURSIVE.LSTM-LAYER",
            .forgetGateNeurons =
            GetSaveInfoOfNeurologicalComponent
            (LSTM_recursiveLayer.value()->forgetGate).denseNeurons,
            .inputGateNeurons =
            GetSaveInfoOfNeurologicalComponent
            (LSTM_recursiveLayer.value()->inputGate).denseNeurons,
            .candidateMemoryGateNeurons =
            GetSaveInfoOfNeurologicalComponent
            (LSTM_recursiveLayer.value()->candidateMemoryGate).denseNeurons,
            .outputGateNeurons =
            GetSaveInfoOfNeurologicalComponent
            (LSTM_recursiveLayer.value()->outputGate).denseNeurons,
        };
    } else if (auto convolutionalLayer = CheckNeurologicalComponent<ConvolutionalLayer>(component)) {
        // save data from convolutional layer
        ConvolutionalLayerSaveInfo convolutionalFilters;

        for (int i = 0; i < convolutionalLayer.value()->filters.size(); i++) {
            const ConvolutionalFilter& filter = convolutionalLayer.value()->filters[i];

            vector<KernelSaveInfo> kernelsSaveInfo;
            for (int kI = 0; kI < filter.kernels.size(); kI++) {
                const Kernel& kernel = filter.kernels[kI];

                KernelSaveInfo kernelSaveInfo;
                for (int _y = 0; _y < kernel.size(); _y++) {
                    const vector<NeurologicalParameter>& kernelWeightsRow = kernel[_y];

                    vector<NeurologicalParameterSaveInfo> kernelRowSaveInfo;
                    for (int _x = 0; _x < kernelWeightsRow.size(); _x++) {
                        kernelRowSaveInfo.push_back(GetParameterSaveInfo(kernelWeightsRow[_x]));
                    }
                    kernelSaveInfo.push_back(kernelRowSaveInfo);
                }
                kernelsSaveInfo.push_back(kernelSaveInfo);
            }
            NeurologicalParameterSaveInfo biasSaveInfo = GetParameterSaveInfo(filter.bias);

            convolutionalFilters.push_back({
                .kernels = kernelsSaveInfo,
                .bias = biasSaveInfo,
            });
        }

        saveInfo = {
            .componentType = "CONVOLUTIONAL-LAYER",
            .convolutionalFilters = convolutionalFilters,
        };
    } else if (auto chain = CheckNeurologicalComponent<NeurologicalComponentsChain>(component)) {
        // save data from neurological components chain
        vector<NeurologicalComponentSaveInfo> componentsChain;
        for (int i = 0; i < chain.value()->chain.size(); i++) {
            componentsChain.push_back(GetSaveInfoOfNeurologicalComponent(chain.value()->chain[i]));
        }

        saveInfo = {
            .componentType = "CHAIN",
            .chain = componentsChain,
        };
    }
    // activation functions
    else if (auto _relu = CheckNeurologicalComponent<ReLU>(component)) {
        saveInfo = {
            .componentType = "AF-ReLU",
        };
    } else if (auto _tanh = CheckNeurologicalComponent<Tanh>(component)) {
        saveInfo = {
            .componentType = "AF-Tanh",
        };
    } else if (auto _sigmoid = CheckNeurologicalComponent<Sigmoid>(component)) {
        saveInfo = {
            .componentType = "AF-Sigmoid",
        };
    } else if (auto _softmax = CheckNeurologicalComponent<Softmax>(component)) {
        saveInfo = {
            .componentType = "AF-Softmax",
        };
    }
    // extras
    else if (auto _convolutionalPool = CheckNeurologicalComponent<ConvolutionalPool>(component)) {
        saveInfo = {
            .componentType = "CONVOLUTIONAL-POOLER",
        };
    } 
    else if (auto _channelsFlattener = CheckNeurologicalComponent<ChannelsFlattener>(component)) {
        saveInfo = {
            .componentType = "CHANNELS-FLATTENER",
        };
    }
    else if (auto _channelsFlattener = CheckNeurologicalComponent<ChannelsSpreader>(component)) {
        saveInfo = {
            .componentType = "CHANNELS-SPREADER",
        };
    }
    
    return saveInfo;
}

void SetSaveInfoOfNeurologicalParameter(NeurologicalParameter& parameter, const NeurologicalParameterSaveInfo& parameterSaveinfo) {
    parameter.value = parameterSaveinfo.value;

    if (parameterSaveinfo.optimizationInfo.has_value()) {
        const NeurologicalParameterOptimizationSaveInfo& _optimizationSaveInfo = parameterSaveinfo.optimizationInfo.value();

        if (_optimizationSaveInfo.optimizerType == "ADAM") {
            ADAM_parameterOptimizationInfo optimizationInfo = ADAM_parameterOptimizationInfo();
            optimizationInfo.v = _optimizationSaveInfo.v.value();
            optimizationInfo.m = _optimizationSaveInfo.m.value();
            optimizationInfo.t = _optimizationSaveInfo.t.value();
            parameter.optimizationInfo = optimizationInfo;
        }
    }
}

void SetSaveInfoOfNeurologicalComponent(INeurologicalComponentPTR component, const NeurologicalComponentSaveInfo& saveInfo) {
    if (auto denseLayer = CheckNeurologicalComponent<DenseLayer>(component)) {
        // load dense layer data
        if (saveInfo.componentType != "DENSE-LAYER") return;
        
        for (int i = 0; i < saveInfo.denseNeurons.value().size(); i++) {
            auto neuronSaveInfo = saveInfo.denseNeurons.value()[i];
            SetSaveInfoOfNeurologicalParameter(
                denseLayer.value()->neurons[i].bias, neuronSaveInfo.bias);

            for (int j = 0; j < neuronSaveInfo.weights.size(); j++) {
                SetSaveInfoOfNeurologicalParameter(
                    denseLayer.value()->neurons[i].weights[j], neuronSaveInfo.weights[j]);
            }
        }
    } else if (auto LSTM_recursiveLayer = CheckNeurologicalComponent<LSTM_RecursiveLayer>(component)) {
        // load lstm recursive layer data
        if (saveInfo.componentType != "RECURSIVE.LSTM-LAYER") return;
        
        SetSaveInfoOfNeurologicalComponent(LSTM_recursiveLayer.value()->forgetGate
        , { .componentType = "DENSE-LAYER", .denseNeurons = saveInfo.forgetGateNeurons.value() });
        SetSaveInfoOfNeurologicalComponent(LSTM_recursiveLayer.value()->inputGate
        , { .componentType = "DENSE-LAYER", .denseNeurons = saveInfo.inputGateNeurons.value() });
        SetSaveInfoOfNeurologicalComponent(LSTM_recursiveLayer.value()->candidateMemoryGate
        , { .componentType = "DENSE-LAYER", .denseNeurons = saveInfo.candidateMemoryGateNeurons.value() });
        SetSaveInfoOfNeurologicalComponent(LSTM_recursiveLayer.value()->outputGate
        , { .componentType = "DENSE-LAYER", .denseNeurons = saveInfo.outputGateNeurons.value() });
    } else if (auto convolutionalLayer = CheckNeurologicalComponent<ConvolutionalLayer>(component)) {
        // load convolutional layer data
        if (saveInfo.componentType != "CONVOLUTIONAL-LAYER") return;

        auto filtersSaveInfo = saveInfo.convolutionalFilters;

        for (int i = 0; i < filtersSaveInfo.value().size(); i++) {
            auto filterSaveInfo = filtersSaveInfo.value()[i];
            
            SetSaveInfoOfNeurologicalParameter(convolutionalLayer.value()->filters[i].bias
            , filterSaveInfo.bias);

            for (int kI = 0; kI < filterSaveInfo.kernels.size(); kI++) {
                auto kernelSaveInfo = filterSaveInfo.kernels[kI];

                Kernel& kernel = convolutionalLayer.value()->filters[i].kernels[kI];
                for (int _y = 0; _y < kernel.size(); _y++) {
                    auto kernelRowSaveInfo = kernelSaveInfo[_y];

                    vector<NeurologicalParameter>& kernelWeightsRow = kernel[_y];

                    for (int _x = 0; _x < kernelWeightsRow.size(); _x++) {
                        SetSaveInfoOfNeurologicalParameter(kernelWeightsRow[_x], kernelRowSaveInfo[_x]);
                    }
                }
            }
        }
    } else if (auto chain = CheckNeurologicalComponent<NeurologicalComponentsChain>(component)) {
        // load neurological components chain data
        if (saveInfo.componentType != "CHAIN") return;

        for (int i = 0; i < chain.value()->chain.size(); i++) {
            SetSaveInfoOfNeurologicalComponent(chain.value()->chain[i], saveInfo.chain.value()[i]);
        }
    }
}

NeurologicalComponentSaveInfo ReadNeurologicalStructure(const string& structureId) {
    INeurologicalComponentPTR structure = neurologicalStructures[structureId];

    return GetSaveInfoOfNeurologicalComponent(structure);
}

void WriteNeurologicalStructure(const string& structureId, const NeurologicalComponentSaveInfo& saveInfo) {
    INeurologicalComponentPTR structure = neurologicalStructures[structureId];

    SetSaveInfoOfNeurologicalComponent(structure, saveInfo);
}

INeurologicalComponentPTR BuildNeurologicalComponent(const NeurologicalComponentBuildInfo& buildInfo) {
    INeurologicalComponentPTR component;
    
    if (buildInfo.componentType == "DENSE-LAYER") {
        component = make_shared<DenseLayer>(buildInfo.inputSize.value(), buildInfo.outputSize.value());
    }
    else if (buildInfo.componentType == "RECURSIVE.LSTM-LAYER") {
        component = make_shared<LSTM_RecursiveLayer>(buildInfo.inputSize.value(), buildInfo.outputSize.value());
    }
    else if (buildInfo.componentType == "CONVOLUTIONAL-LAYER") {
        vector<string> splitedKernelSizeString = splitStringByString(buildInfo.kernelSize.value(), "x");
        component = make_shared<ConvolutionalLayer>(
            buildInfo.inputSize.value(), buildInfo.outputSize.value(),
            buildInfo.padding.value(), buildInfo.stride.value(),
            stoi(splitedKernelSizeString[0]), stoi(splitedKernelSizeString[1])
        );
    }
    else if (buildInfo.componentType == "CONVOLUTIONAL-POOLER") {
        vector<string> splitedPoolSizeString = splitStringByString(buildInfo.poolSize.value(), "x");
        component = make_shared<ConvolutionalPool>(
            buildInfo.poolingType.value(),
            stoi(splitedPoolSizeString[0]), stoi(splitedPoolSizeString[1]),
            buildInfo.stride.value()
        );
    }
    else if (buildInfo.componentType == "CHANNELS-FLATTENER") {
        component = make_shared<ChannelsFlattener>();
    }
    else if (buildInfo.componentType == "CHANNELS-SPREADER") {
        component = make_shared<ChannelsSpreader>(buildInfo.expectedChannelsAmount.value()
            , buildInfo.expectedChannelsWidth.value()
            , buildInfo.expectedChannelsHeight.value());
    }
    else if (buildInfo.componentType == "CHAIN") {
        vector<INeurologicalComponentPTR> _chain;
        
        for (int i = 0; i < buildInfo.chain.value().size(); i++) {
            _chain.push_back(BuildNeurologicalComponent(buildInfo.chain.value()[i]));
        }

        component = make_shared<NeurologicalComponentsChain>(_chain);
    }
    else if (buildInfo.componentType == "AF-ReLU") {
        component = make_shared<ReLU>(buildInfo._negativeLeakage.value());
    }
    else if (buildInfo.componentType == "AF-Tanh") {
        component = make_shared<Tanh>();
    }
    else if (buildInfo.componentType == "AF-Sigmoid") {
        component = make_shared<Sigmoid>();
    }
    else if (buildInfo.componentType == "AF-Softmax") {
        component = make_shared<Softmax>(buildInfo._ignoreJacobian.value());
    } else {
        cout << "NO NEUROLOGICAL STRUCTURE BUILD AVAILABLE FOR: " << buildInfo.componentType << endl;
    }

    return component;
}

optional<OptimizerPTR> TryBuildNeurologicalOptimizer(optional<OptimizerBuildInfo> buildInfo) {
    if (!buildInfo.has_value()) {
        return nullopt;
    }

    OptimizerPTR optimizer;

    if (buildInfo.value().optimizerType == "ADAM") {
        optimizer = make_shared<ADAM_Optimizer>(
            buildInfo.value().epsilon.value(),
            buildInfo.value().beta1.value(),
            buildInfo.value().beta2.value());

    } else {
        cout << "NO OPTIMIZER BUILD AVAILABLE FOR: " << buildInfo.value().optimizerType << endl;
    }

    return optimizer;
}

optional<ClippingMethodPTR> TryBuildClippingMethod(optional<ClippingMethodBuildinfo> buildInfo) {
    if (!buildInfo.has_value()) {
        return nullopt;
    }

    ClippingMethodPTR clippingMethod;

    if (buildInfo.value().clippingMethodType == "NORM") {
        clippingMethod = make_shared<NORM_ClippingMethod>(
            buildInfo.value().clip_value.value());
    } else {
        cout << "NO CLIPPING METHOD BUILD AVAILABLE FOR: " << buildInfo.value().clippingMethodType << endl;
    }

    return clippingMethod;
}

string CreateNeurologicalStructure(const NeurologicalComponentBuildInfo& buildInfo) {
    int _index = neurologicalStructureCreationIndex;
    neurologicalStructureCreationIndex+=1;

    string id = "neurological-structure->" + to_string(_index);
    neurologicalStructures[id] = dynamic_pointer_cast<NeurologicalComponentsChain>(BuildNeurologicalComponent(buildInfo));
    return id;
}

void InitializeNeurologicalStructure(const string& structureId, const NeurologicalStructureInitializationInfo& intializationInfo) {
    NeurologicalStructure structure = neurologicalStructures[structureId];
    
    structure->initialize({
        .optimizer = TryBuildNeurologicalOptimizer(intializationInfo.optimizerBuild),
        .clippingMethod = TryBuildClippingMethod(intializationInfo.clippingMethodBuild),
    });
}

NeurologicalPassingValues ActivateNeurologicalStructure(const string& structureId, const NeurologicalPassingValues& input, const string& sampleIndex) {
    NeurologicalStructure structure = neurologicalStructures[structureId];

    NeurologicalPassingValues output = structure->forwardPass(input, sampleIndex);

    return output;
}
NeurologicalPassingValues AdjustNeurologicalStructure(const string& structureId, const NeurologicalPassingValues& gradient, const string& sampleIndex) {
    NeurologicalStructure structure = neurologicalStructures[structureId];

    return structure->backPropagation(gradient, sampleIndex);
}
void ApplyNeurologicalStructureAdjustments(const string& structureId, const derivativesApplyingInfo& applyingInfo) {
    NeurologicalStructure structure = neurologicalStructures[structureId];

    structure->applyDerivatives(applyingInfo);
}