#include "library/LibraryController.h"

// void ActivateNeurologicalStructureFunc(pybind11::buffer input, pybind11::buffer output, const int& batchSize)
// {
//     const cuda_neurologicalConstantSpan inputSpan = bufferToConstantSpan(input);
//     const cuda_neurologicalSpan outputSpan = bufferToSpan(output);

//     testActivation(inputSpan, outputSpan, batchSize);
// }

PYBIND11_MODULE(cuda_module, _handle)
{
    pybind11::enum_<cuda_NeurologicalComponentTypes>(_handle, "ComponentType")
        .value("COMPONENTS_CHAIN", cuda_NeurologicalComponentTypes::COMPONENTS_CHAIN)
        .value("DENSE_LAYER", cuda_NeurologicalComponentTypes::DENSE_LAYER)
        .value("CONVOLUTIONAL_LAYER", cuda_NeurologicalComponentTypes::CONVOLUTIONAL_LAYER)
        .value("RMSNORM_LAYER", cuda_NeurologicalComponentTypes::RMSNORM_LAYER)
        .value("MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER", cuda_NeurologicalComponentTypes::MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER)
        .value("SCC_POSITIONAL_EMBEDDING_LAYER", cuda_NeurologicalComponentTypes::SCC_POSITIONAL_EMBEDDING_LAYER);

    pybind11::class_<BaseNeurologicalComponent__BuildInfo,
               std::shared_ptr<BaseNeurologicalComponent__BuildInfo>>(_handle, "BaseBuildInfo")
        .def_readwrite("componentType", &BaseNeurologicalComponent__BuildInfo::componentType)
        .def_readwrite("freeze",        &BaseNeurologicalComponent__BuildInfo::freeze)
        .def_readwrite("residual",      &BaseNeurologicalComponent__BuildInfo::residual);

    pybind11::class_<RMSNormLayer__BuildInfo,
            BaseNeurologicalComponent__BuildInfo,
            std::shared_ptr<RMSNormLayer__BuildInfo>>(_handle, "RMSNormLayerBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("tensorSize", &RMSNormLayer__BuildInfo::tensorSize);
    pybind11::class_<DenseLayer__BuildInfo,
               BaseNeurologicalComponent__BuildInfo,
               std::shared_ptr<DenseLayer__BuildInfo>>(_handle, "DenseLayerBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("inputSize", &DenseLayer__BuildInfo::inputSize)
        .def_readwrite("outputSize", &DenseLayer__BuildInfo::outputSize)
        .def_readwrite("maxDropout", &DenseLayer__BuildInfo::maxDropout)
        .def_readwrite("activation", &DenseLayer__BuildInfo::activation);

    pybind11::class_<ConvolutionalLayer__BuildInfo,
               BaseNeurologicalComponent__BuildInfo,
               std::shared_ptr<ConvolutionalLayer__BuildInfo>>(_handle, "ConvolutionalLayerBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("outputChannels", &ConvolutionalLayer__BuildInfo::outputChannels)
        .def_readwrite("inputChannels",  &ConvolutionalLayer__BuildInfo::inputChannels)
        .def_readwrite("slideWidth",     &ConvolutionalLayer__BuildInfo::slideWidth)
        .def_readwrite("slideHeight",    &ConvolutionalLayer__BuildInfo::slideHeight)
        .def_readwrite("inputWidth",     &ConvolutionalLayer__BuildInfo::inputWidth)
        .def_readwrite("inputHeight",    &ConvolutionalLayer__BuildInfo::inputHeight)
        .def_readwrite("stride",         &ConvolutionalLayer__BuildInfo::stride)
        .def_readwrite("padding",        &ConvolutionalLayer__BuildInfo::padding)
        .def_readwrite("maxDropout",     &ConvolutionalLayer__BuildInfo::maxDropout)
        .def_readwrite("activation",     &ConvolutionalLayer__BuildInfo::activation);

    pybind11::class_<MultiHeadMaskedSelfAttentionLayer__BuildInfo,
               BaseNeurologicalComponent__BuildInfo,
               std::shared_ptr<MultiHeadMaskedSelfAttentionLayer__BuildInfo>>(_handle, "MultiHeadMaskedSelfAttentionLayerBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("inputSize", &MultiHeadMaskedSelfAttentionLayer__BuildInfo::inputSize)
        .def_readwrite("headOutputSize", &MultiHeadMaskedSelfAttentionLayer__BuildInfo::headOutputSize)
        .def_readwrite("headsAmount", &MultiHeadMaskedSelfAttentionLayer__BuildInfo::headsAmount);

    pybind11::class_<SCCPositionalEmbeddingLayer__BuildInfo,
               BaseNeurologicalComponent__BuildInfo,
               std::shared_ptr<SCCPositionalEmbeddingLayer__BuildInfo>>(_handle, "SCCPositionalEmbeddingLayerBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("inputSize", &SCCPositionalEmbeddingLayer__BuildInfo::inputSize)
        .def_readwrite("outputSize", &SCCPositionalEmbeddingLayer__BuildInfo::outputSize);

    pybind11::class_<ComponentsChain__BuildInfo,
               BaseNeurologicalComponent__BuildInfo,
               std::shared_ptr<ComponentsChain__BuildInfo>>(_handle, "ComponentsChainBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("components", &ComponentsChain__BuildInfo::components);

    // Activation enum
    pybind11::enum_<cuda_ActivationFunctionTypes>(_handle, "ActivationType")
        .value("LINEAR", cuda_ActivationFunctionTypes::LINEAR)
        .value("ReLU", cuda_ActivationFunctionTypes::ReLU);

    // Base Activation
    pybind11::class_<BaseActivationFunction__BuildInfo,
            std::shared_ptr<BaseActivationFunction__BuildInfo>>(_handle, "BaseActivationBuildInfo")
        .def_readwrite("componentType", &BaseActivationFunction__BuildInfo::componentType);

    // ReLU
    pybind11::class_<ReLU__BuildInfo,
            BaseActivationFunction__BuildInfo,
            std::shared_ptr<ReLU__BuildInfo>>(_handle, "ReLUActivationBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("fadeMultiplier", &ReLU__BuildInfo::fadeMultiplier);

    // Optimizer enum
    pybind11::enum_<cuda_OptimizerTypes>(_handle, "OptimizerType")
        .value("DEFAULT", cuda_OptimizerTypes::DEFAULT)
        .value("ADAM", cuda_OptimizerTypes::ADAM);

    // Base Optimizer
    pybind11::class_<BaseOptimizer__BuildInfo,
            std::shared_ptr<BaseOptimizer__BuildInfo>>(_handle, "BaseOptimizerBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("optimizerType", &BaseOptimizer__BuildInfo::optimizerType);

    // ADAM
    pybind11::class_<ADAM__BuildInfo,
            BaseOptimizer__BuildInfo,
            std::shared_ptr<ADAM__BuildInfo>>(_handle, "ADAMOptimizerBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("beta1", &ADAM__BuildInfo::beta1)
        .def_readwrite("beta2", &ADAM__BuildInfo::beta2)
        .def_readwrite("epsilon", &ADAM__BuildInfo::epsilon);

    // ===== Save Info =====
    pybind11::class_<BaseNeurologicalComponent__SaveInfo,
               std::shared_ptr<BaseNeurologicalComponent__SaveInfo>>(_handle, "BaseSaveInfo")
        .def_readwrite("componentType", &BaseNeurologicalComponent__SaveInfo::componentType);

    pybind11::class_<RMSNormLayer__SaveInfo,
               BaseNeurologicalComponent__SaveInfo,
               std::shared_ptr<RMSNormLayer__SaveInfo>>(_handle, "RMSNormLayerSaveInfo")
        .def(pybind11::init<>())
        .def_readwrite("gamma", &RMSNormLayer__SaveInfo::gamma);

    pybind11::class_<DenseLayer__SaveInfo,
               BaseNeurologicalComponent__SaveInfo,
               std::shared_ptr<DenseLayer__SaveInfo>>(_handle, "DenseLayerSaveInfo")
        .def(pybind11::init<>())
        .def_readwrite("weights", &DenseLayer__SaveInfo::weights)
        .def_readwrite("biases",  &DenseLayer__SaveInfo::biases);

    pybind11::class_<ConvolutionalLayer__SaveInfo,
               BaseNeurologicalComponent__SaveInfo,
               std::shared_ptr<ConvolutionalLayer__SaveInfo>>(_handle, "ConvolutionalLayerSaveInfo")
        .def(pybind11::init<>())
        .def_readwrite("weights", &ConvolutionalLayer__SaveInfo::weights)
        .def_readwrite("biases",  &ConvolutionalLayer__SaveInfo::biases);

    pybind11::class_<MultiHeadMaskedSelfAttentionLayer__SaveInfo,
               BaseNeurologicalComponent__SaveInfo,
               std::shared_ptr<MultiHeadMaskedSelfAttentionLayer__SaveInfo>>(_handle, "MultiHeadMaskedSelfAttentionLayerSaveInfo")
        .def(pybind11::init<>())
        .def_readwrite("weights", &MultiHeadMaskedSelfAttentionLayer__SaveInfo::weights);

    pybind11::class_<SCCPositionalEmbeddingLayer__SaveInfo,
               BaseNeurologicalComponent__SaveInfo,
               std::shared_ptr<SCCPositionalEmbeddingLayer__SaveInfo>>(_handle, "SCCPositionalEmbeddingLayerSaveInfo")
        .def(pybind11::init<>())
        .def_readwrite("weights", &SCCPositionalEmbeddingLayer__SaveInfo::weights);

    pybind11::class_<ComponentsChain__SaveInfo,
               BaseNeurologicalComponent__SaveInfo,
               std::shared_ptr<ComponentsChain__SaveInfo>>(_handle, "ComponentsChainSaveInfo")
        .def(pybind11::init<>())
        .def_readwrite("components", &ComponentsChain__SaveInfo::components);

    _handle.def("BuildModel", &BuildModel);
    
    _handle.def("InitializeModel", &InitializeModel);

    _handle.def("InferModel",    &InferModel);
    _handle.def("ClearModelInference",    &ClearModelInference);
    _handle.def("ActivateModel", &ActivateModel);
    _handle.def("AdjustModel", &AdjustModel);
    
    _handle.def("UpdateModel", &UpdateModel);
    
    _handle.def("ExtractModel", &ExtractModel);
    _handle.def("InsertModel",  &InsertModel);

    _handle.def("MSE", &MSE);
    
    _handle.def("softmax", &softmax);
    // _handle.def("ActivateNeurologicalStructure", &ActivateNeurologicalStructureFunc);
}