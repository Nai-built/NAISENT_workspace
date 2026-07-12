#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../cpp/cppCode.h"
#include "../cpp/NeurologicalStructureLogic.h"
#include "../cpp/AI/LossFunctions.h"

PYBIND11_MODULE(simple_module, _handle)
{
    _handle.def("multiply", &multiply);
    _handle.def("add", &add);
    _handle.def("nextInteger", &nextInteger);

    pybind11::class_<NeurologicalParameterOptimizationSaveInfo>(_handle, "NeurologicalParameterOptimizationSaveInfo")
        .def(pybind11::init<>())
        .def_readwrite("optimizerType", &NeurologicalParameterOptimizationSaveInfo::optimizerType)

        .def_readwrite("v", &NeurologicalParameterOptimizationSaveInfo::v)
        .def_readwrite("m", &NeurologicalParameterOptimizationSaveInfo::m)
        .def_readwrite("t", &NeurologicalParameterOptimizationSaveInfo::t);

    pybind11::class_<NeurologicalParameterSaveInfo>(_handle, "NeurologicalParameterSaveInfo")
        .def(pybind11::init<>())
        .def_readwrite("value", &NeurologicalParameterSaveInfo::value)
        .def_readwrite("optimizationInfo", &NeurologicalParameterSaveInfo::optimizationInfo);

    pybind11::class_<DenseNeuronSaveInfo>(_handle, "DenseNeuronSaveInfo")
        .def(pybind11::init<>())
        .def_readwrite("weights", &DenseNeuronSaveInfo::weights)
        .def_readwrite("bias", &DenseNeuronSaveInfo::bias);

    pybind11::class_<ConvolutionalFilterSaveInfo>(_handle, "ConvolutionalFilterSaveInfo")
        .def(pybind11::init<>())
        .def_readwrite("kernels", &ConvolutionalFilterSaveInfo::kernels)
        .def_readwrite("bias", &ConvolutionalFilterSaveInfo::bias);

    pybind11::class_<NeurologicalComponentSaveInfo>(_handle, "NeurologicalComponentSaveInfo")
        .def(pybind11::init<>())
        .def_readwrite("componentType", &NeurologicalComponentSaveInfo::componentType)

        .def_readwrite("denseNeurons", &NeurologicalComponentSaveInfo::denseNeurons)

        .def_readwrite("forgetGateNeurons", &NeurologicalComponentSaveInfo::forgetGateNeurons)
        .def_readwrite("inputGateNeurons", &NeurologicalComponentSaveInfo::inputGateNeurons)
        .def_readwrite("candidateMemoryGateNeurons", &NeurologicalComponentSaveInfo::candidateMemoryGateNeurons)
        .def_readwrite("outputGateNeurons", &NeurologicalComponentSaveInfo::outputGateNeurons)

        .def_readwrite("convolutionalFilters", &NeurologicalComponentSaveInfo::convolutionalFilters)

        .def_readwrite("chain", &NeurologicalComponentSaveInfo::chain);

    pybind11::class_<NeurologicalComponentBuildInfo>(_handle, "NeurologicalComponentBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("componentType", &NeurologicalComponentBuildInfo::componentType)

        .def_readwrite("inputSize", &NeurologicalComponentBuildInfo::inputSize)
        .def_readwrite("outputSize", &NeurologicalComponentBuildInfo::outputSize)

        .def_readwrite("stride", &NeurologicalComponentBuildInfo::stride)

        .def_readwrite("padding", &NeurologicalComponentBuildInfo::padding)
        .def_readwrite("kernelSize", &NeurologicalComponentBuildInfo::kernelSize)
        
        .def_readwrite("poolSize", &NeurologicalComponentBuildInfo::poolSize)
        .def_readwrite("poolingType", &NeurologicalComponentBuildInfo::poolingType)
        
        .def_readwrite("expectedChannelsAmount", &NeurologicalComponentBuildInfo::expectedChannelsAmount)
        .def_readwrite("expectedChannelsWidth", &NeurologicalComponentBuildInfo::expectedChannelsWidth)
        .def_readwrite("expectedChannelsHeight", &NeurologicalComponentBuildInfo::expectedChannelsHeight)

        .def_readwrite("chain", &NeurologicalComponentBuildInfo::chain)

        .def_readwrite("_negativeLeakage", &NeurologicalComponentBuildInfo::_negativeLeakage)
        

        .def_readwrite("_ignoreJacobian", &NeurologicalComponentBuildInfo::_ignoreJacobian);

    pybind11::class_<OptimizerBuildInfo>(_handle, "OptimizerBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("optimizerType", &OptimizerBuildInfo::optimizerType)
        
        .def_readwrite("epsilon", &OptimizerBuildInfo::epsilon)
        .def_readwrite("beta1", &OptimizerBuildInfo::beta1)
        .def_readwrite("beta2", &OptimizerBuildInfo::beta2);
        
    pybind11::class_<ClippingMethodBuildinfo>(_handle, "ClippingMethodBuildinfo")
        .def(pybind11::init<>())
        .def_readwrite("clippingMethodType", &ClippingMethodBuildinfo::clippingMethodType)
        
        .def_readwrite("clip_value", &ClippingMethodBuildinfo::clip_value);

    pybind11::class_<NeurologicalStructureInitializationInfo>(_handle, "NeurologicalStructureInitializationInfo")
        .def(pybind11::init<>())
        .def_readwrite("optimizerBuild", &NeurologicalStructureInitializationInfo::optimizerBuild)
        .def_readwrite("clippingMethodBuild", &NeurologicalStructureInitializationInfo::clippingMethodBuild);

    pybind11::class_<derivativesApplyingInfo>(_handle, "derivativesApplyingInfo")
        .def(pybind11::init<>())
        .def_readwrite("learningRate", &derivativesApplyingInfo::learningRate)
        .def_readwrite("averagingDivisor", &derivativesApplyingInfo::averagingDivisor);

    _handle.def("ReadNeurologicalStructure", &ReadNeurologicalStructure);
    _handle.def("WriteNeurologicalStructure", &WriteNeurologicalStructure);

    _handle.def("CreateNeurologicalStructure", &CreateNeurologicalStructure);
    _handle.def("InitializeNeurologicalStructure", &InitializeNeurologicalStructure);
    _handle.def("ActivateNeurologicalStructure", &ActivateNeurologicalStructure);
    _handle.def("AdjustNeurologicalStructure", &AdjustNeurologicalStructure);
    _handle.def("ApplyNeurologicalStructureAdjustments", &ApplyNeurologicalStructureAdjustments);

    
    _handle.def("zero_loss_gradient", &zero_loss_gradient);
    _handle.def("MSE_loss_gradient", &MSE_loss_gradient);
    _handle.def("MSE_loss_value", &MSE_loss_value);
    _handle.def("KL_divergence_gradient", &KL_divergence_gradient);
    _handle.def("KL_divergence_value", &KL_divergence_value);
}
