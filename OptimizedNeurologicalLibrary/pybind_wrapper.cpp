#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "library/LibraryController.h"
#include "library/LossFunctions.h"

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

    if (bufferInfo.format != pybind11::format_descriptor<neurologicalValue>::format())
        throw std::runtime_error("Expected neurological value buffer");

    if (bufferInfo.strides[0] != sizeof(neurologicalValue))
        throw std::runtime_error("Buffer is not contiguous");

    neurologicalValue* bufferData = static_cast<neurologicalValue*>(bufferInfo.ptr);
    size_t bufferSize = static_cast<size_t>(bufferInfo.shape[0]);

    return neurologicalConstantSpan(bufferData, bufferSize);
}
inline neurologicalSpan bufferToSpan(pybind11::buffer buffer) {
    pybind11::buffer_info bufferInfo = buffer.request();

    if (bufferInfo.readonly)
        throw std::runtime_error("Buffer is read-only");

    if (bufferInfo.ndim != 1)
        throw std::runtime_error("Expected 1D buffer");

    if (bufferInfo.format != pybind11::format_descriptor<neurologicalValue>::format())
        throw std::runtime_error("Expected neurological value buffer");

    if (bufferInfo.strides[0] != sizeof(neurologicalValue))
        throw std::runtime_error("Buffer is not contiguous");

    neurologicalValue* bufferData = static_cast<neurologicalValue*>(bufferInfo.ptr);
    size_t bufferSize = static_cast<size_t>(bufferInfo.shape[0]);

    return neurologicalSpan(bufferData, bufferSize);
}

void ActivateNeurologicalStructureFunc(std::string id, pybind11::buffer input, pybind11::buffer output, const int& batchSize
    , pybind11::buffer seriesLengths, const int& totalSamples)
{
    const neurologicalConstantSpan inputSpan = bufferToConstantSpan(input);
    const neurologicalSpan outputSpan = bufferToSpan(output);

    const lengths seriesLengthsSpan = bufferToConstantLengths(seriesLengths);

    ActivateNeurologicalStructure(id, inputSpan, outputSpan, batchSize, seriesLengthsSpan, totalSamples);
}

void AdjustNeurologicalStructureFunc(std::string id, pybind11::buffer propagation, pybind11::buffer inputPropagation, const neurologicalValue& learnRate, const int& batchSize
    , pybind11::buffer seriesLengths, const int& totalSamples)
{
    const neurologicalSpan propagationSpan = bufferToSpan(propagation);
    const neurologicalSpan inputPropagationSpan = bufferToSpan(inputPropagation);
    
    const lengths seriesLengthsSpan = bufferToConstantLengths(seriesLengths);

    AdjustNeurologicalStructure(id, propagationSpan, inputPropagationSpan, learnRate, batchSize, seriesLengthsSpan, totalSamples);
}

void MSE(pybind11::buffer prediction, pybind11::buffer correction, pybind11::buffer propagation) {
    const neurologicalConstantSpan predictionSpan = bufferToConstantSpan(prediction);
    const neurologicalConstantSpan correctionSpan = bufferToConstantSpan(correction);
    const neurologicalSpan propagationSpan = bufferToSpan(propagation);
    MSE_loss_propagation(predictionSpan, correctionSpan, propagationSpan);
}

void softmax_F(pybind11::buffer values, const int& eachSize) {
    const neurologicalSpan valuesSpan = bufferToSpan(values);
    softmax(valuesSpan, eachSize);
}

PYBIND11_MODULE(simple_module, _handle)
{
    pybind11::enum_<ChannelsPoolerTypes>(_handle, "ChannelsPoolerTypes")
        .value("NONE", ChannelsPoolerTypes::NONE)

        .value("MAX", ChannelsPoolerTypes::MAX)
        .value("AVERAGE", ChannelsPoolerTypes::AVERAGE)
        .value("MIN", ChannelsPoolerTypes::MIN);

    pybind11::class_<BaseChannelsPoolerBuildInfo, std::shared_ptr<BaseChannelsPoolerBuildInfo>>(_handle, "BaseChannelsPoolerBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("poolerType", &BaseChannelsPoolerBuildInfo::poolerType)

        .def_readwrite("poolingSize", &BaseChannelsPoolerBuildInfo::poolingSize)
        .def_readwrite("stride", &BaseChannelsPoolerBuildInfo::stride);

    pybind11::class_<MaxPool__BuildInfo, std::shared_ptr<MaxPool__BuildInfo>, BaseChannelsPoolerBuildInfo>(_handle, "MaxPool__BuildInfo")
        .def(pybind11::init<>());
    pybind11::class_<AveragePool__BuildInfo, std::shared_ptr<AveragePool__BuildInfo>, BaseChannelsPoolerBuildInfo>(_handle, "AveragePool__BuildInfo")
        .def(pybind11::init<>());
    pybind11::class_<MinPool__BuildInfo, std::shared_ptr<MinPool__BuildInfo>, BaseChannelsPoolerBuildInfo>(_handle, "MinPool__BuildInfo")
        .def(pybind11::init<>());

    pybind11::enum_<ActivationFunctionTypes>(_handle, "ActivationFunctionTypes")
        .value("LINEAR", ActivationFunctionTypes::LINEAR)
        .value("RELU", ActivationFunctionTypes::RELU);

    pybind11::class_<BaseActivationFunctionBuildInfo, std::shared_ptr<BaseActivationFunctionBuildInfo>>(_handle, "BaseActivationFunctionBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("activationType", &BaseActivationFunctionBuildInfo::activationType);
        
    pybind11::class_<ReLU__BuildInfo, std::shared_ptr<ReLU__BuildInfo>, BaseActivationFunctionBuildInfo>(_handle, "ReLU__BuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("leakage", &ReLU__BuildInfo::leakage);

    pybind11::enum_<NeurologicalComponentTypes>(_handle, "NeurologicalComponentTypes")
        .value("COMPONENTS_CHAIN", NeurologicalComponentTypes::COMPONENTS_CHAIN)
        .value("DENSE_LAYER", NeurologicalComponentTypes::DENSE_LAYER)
        .value("CONVOLUTIONAL_LAYER", NeurologicalComponentTypes::CONVOLUTIONAL_LAYER)
        .value("RECURSIVE_LSTM", NeurologicalComponentTypes::RECURSIVE_LSTM)
        .value("DECODER_ONLY_TRANSFORMER", NeurologicalComponentTypes::DECODER_ONLY_TRANSFORMER);

    pybind11::class_<BaseNeurologicalComponentBuildInfo, std::shared_ptr<BaseNeurologicalComponentBuildInfo>>(_handle, "BaseNeurologicalComponentBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("componentType", &BaseNeurologicalComponentBuildInfo::componentType);
        
    pybind11::class_<DenseLayerBuildInfo, std::shared_ptr<DenseLayerBuildInfo>, BaseNeurologicalComponentBuildInfo>(_handle, "DenseLayerBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("inputSize", &DenseLayerBuildInfo::inputSize)
        .def_readwrite("outputSize", &DenseLayerBuildInfo::outputSize)
        .def_readwrite("activationFunction", &DenseLayerBuildInfo::activationFunction);

    pybind11::class_<ConvolutionalLayerBuildInfo, std::shared_ptr<ConvolutionalLayerBuildInfo>, BaseNeurologicalComponentBuildInfo>(_handle, "ConvolutionalLayerBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("inputChannels", &ConvolutionalLayerBuildInfo::inputChannels)
        .def_readwrite("outputChannels", &ConvolutionalLayerBuildInfo::outputChannels)

        .def_readwrite("stride", &ConvolutionalLayerBuildInfo::stride)
        .def_readwrite("padding", &ConvolutionalLayerBuildInfo::padding)

        .def_readwrite("kernelSize", &ConvolutionalLayerBuildInfo::kernelSize)
        .def_readwrite("inputChannelSize", &ConvolutionalLayerBuildInfo::inputChannelSize)

        .def_readwrite("activationFunction", &ConvolutionalLayerBuildInfo::activationFunction)
        .def_readwrite("channelsPooler", &ConvolutionalLayerBuildInfo::channelsPooler);

    pybind11::class_<LSTM_RecursiveLayerBuildInfo, std::shared_ptr<LSTM_RecursiveLayerBuildInfo>, BaseNeurologicalComponentBuildInfo>(_handle, "LSTM_RecursiveLayerBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("inputSize", &LSTM_RecursiveLayerBuildInfo::inputSize)
        .def_readwrite("outputSize", &LSTM_RecursiveLayerBuildInfo::outputSize)
        .def_readwrite("castBeyond", &LSTM_RecursiveLayerBuildInfo::castBeyond);

    pybind11::class_<ComponentsChainBuildInfo, std::shared_ptr<ComponentsChainBuildInfo>, BaseNeurologicalComponentBuildInfo>(_handle, "ComponentsChainBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("componentsChainBuild", &ComponentsChainBuildInfo::componentsChainBuild);
        
    // TRANSFORMERS BUILDS
    pybind11::class_<HiddenFFL_BuildInfo, std::shared_ptr<HiddenFFL_BuildInfo>>(_handle, "HiddenFFL_BuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("hidden_size", &HiddenFFL_BuildInfo::hidden_size)
        .def_readwrite("activationFunction", &HiddenFFL_BuildInfo::activationFunction);

    pybind11::class_<TransformerStack_BuildInfo, std::shared_ptr<TransformerStack_BuildInfo>>(_handle, "TransformerStack_BuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("attention_heads", &TransformerStack_BuildInfo::attention_heads)
        .def_readwrite("ffn", &TransformerStack_BuildInfo::ffn);
        
    pybind11::class_<DecoderOnly_TransformerBuildInfo, std::shared_ptr<DecoderOnly_TransformerBuildInfo>, BaseNeurologicalComponentBuildInfo>(_handle, "DecoderOnly_TransformerBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("inputTokens", &DecoderOnly_TransformerBuildInfo::inputTokens)
        .def_readwrite("outputTokens", &DecoderOnly_TransformerBuildInfo::outputTokens)
        .def_readwrite("encodingSize", &DecoderOnly_TransformerBuildInfo::encodingSize)

        .def_readwrite("stacksBuild", &DecoderOnly_TransformerBuildInfo::stacksBuild);

    // SAVE INFOs
    pybind11::class_<BaseNeurologicalComponentSaveInfo, std::shared_ptr<BaseNeurologicalComponentSaveInfo>>(_handle, "BaseNeurologicalComponentSaveInfo")
        .def(pybind11::init<>())
        .def_readwrite("saveType", &BaseNeurologicalComponentSaveInfo::saveType);

    pybind11::class_<CommonLayerSaveInfo, std::shared_ptr<CommonLayerSaveInfo>, BaseNeurologicalComponentSaveInfo>(_handle, "CommonLayerSaveInfo")
        .def(pybind11::init<>())
        .def_readwrite("weights", &CommonLayerSaveInfo::weights)
        .def_readwrite("biases", &CommonLayerSaveInfo::biases);

    pybind11::class_<TransformerSaveInfo, std::shared_ptr<TransformerSaveInfo>, BaseNeurologicalComponentSaveInfo>(_handle, "TransformerSaveInfo")
        .def(pybind11::init<>())
        
        .def_readwrite("encodeWeights", &TransformerSaveInfo::encodeWeights)
        
        .def_readwrite("attentionWeights", &TransformerSaveInfo::attentionWeights)
        .def_readwrite("attentionFinalBiases", &TransformerSaveInfo::attentionFinalBiases)
        
        .def_readwrite("normalizationGammas", &TransformerSaveInfo::normalizationGammas)
        
        .def_readwrite("feedForwardNetworksWeights", &TransformerSaveInfo::feedForwardNetworksWeights)
        .def_readwrite("feedForwardNetworksBiases", &TransformerSaveInfo::feedForwardNetworksBiases)
        
        .def_readwrite("decodeWeights", &TransformerSaveInfo::decodeWeights)
        .def_readwrite("decodeBiases", &TransformerSaveInfo::decodeBiases);
        
    pybind11::class_<ComponentsChainSaveInfo, std::shared_ptr<ComponentsChainSaveInfo>, BaseNeurologicalComponentSaveInfo>(_handle, "ComponentsChainSaveInfo")
        .def(pybind11::init<>())
        .def_readwrite("componentsChainSave", &ComponentsChainSaveInfo::componentsChainSave);

    _handle.def("ReadNeurologicalStructure", &ReadNeurologicalStructure);
    _handle.def("WriteNeurologicalStructure", &WriteNeurologicalStructure);

    _handle.def("BuildNeurologicalStructure", &BuildNeurologicalStructure);
    _handle.def("InitializeNeurologicalStructure", &InitializeNeurologicalStructure);

    _handle.def("ActivateNeurologicalStructure", &ActivateNeurologicalStructureFunc);
    _handle.def("AdjustNeurologicalStructure", &AdjustNeurologicalStructureFunc);
    
    _handle.def("MSE", &MSE);
    
    _handle.def("softmax", &softmax_F);
}