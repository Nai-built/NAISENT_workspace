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

    pybind11::class_<AdamParameterOptimizationInfo>(_handle, "AdamParameterOptimizationInfo")
        .def(pybind11::init<>())
        .def_readwrite("v", &AdamParameterOptimizationInfo::v)
        .def_readwrite("m", &AdamParameterOptimizationInfo::m)
        .def_readwrite("t", &AdamParameterOptimizationInfo::t);

    pybind11::class_<NeurologicalParameterSaveInfo>(_handle, "NeurologicalParameterSaveInfo")
        .def(pybind11::init<>())
        .def_readwrite("value", &NeurologicalParameterSaveInfo::value)
        .def_readwrite("optimizationInfo", &NeurologicalParameterSaveInfo::optimizationInfo);

    pybind11::class_<DenseNeuronSaveInfo>(_handle, "DenseNeuronSaveInfo")
        .def(pybind11::init<>())
        .def_readwrite("weights", &DenseNeuronSaveInfo::weights)
        .def_readwrite("bias", &DenseNeuronSaveInfo::bias);

    pybind11::class_<DenseLayerSaveInfo>(_handle, "DenseLayerSaveInfo")
        .def(pybind11::init<>())
        .def_readwrite("denseNeurons", &DenseLayerSaveInfo::denseNeurons);
    pybind11::class_<ChainSaveInfo>(_handle, "ChainSaveInfo")
        .def(pybind11::init<>())
        .def_readwrite("chain", &ChainSaveInfo::chain);

    pybind11::class_<DenseLayerBuildInfo>(_handle, "DenseLayerBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("inputSize", &DenseLayerBuildInfo::inputSize)
        .def_readwrite("outputSize", &DenseLayerBuildInfo::outputSize);
    pybind11::class_<ChainBuildInfo>(_handle, "ChainBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("chain", &ChainBuildInfo::chain);
    pybind11::class_<AF_ReLUBuildInfo>(_handle, "AF_ReLUBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("_negativeLeakage", &AF_ReLUBuildInfo::_negativeLeakage);
    pybind11::class_<AF_TanhBuildInfo>(_handle, "AF_TanhBuildInfo")
        .def(pybind11::init<>());
    pybind11::class_<AF_SoftmaxBuildInfo>(_handle, "AF_SoftmaxBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("_ignoreJacobian", &AF_ReLUBuildInfo::_ignoreJacobian);

    pybind11::class_<AdamOptimizerBuildInfo>(_handle, "AdamOptimizerBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("epsilon", &AdamOptimizerBuildInfo::epsilon)
        .def_readwrite("beta1", &AdamOptimizerBuildInfo::beta1)
        .def_readwrite("beta2", &AdamOptimizerBuildInfo::beta2);
        
    pybind11::class_<NormClippingMethodBuildInfo>(_handle, "NormClippingMethodBuildInfo")
        .def(pybind11::init<>())
        .def_readwrite("clip_value", &NormClippingMethodBuildInfo::clip_value);

    pybind11::class_<NeurologicalStructureInitializationInfo>(_handle, "NeurologicalStructureInitializationInfo")
        .def(pybind11::init<>())
        .def_readwrite("optimizerBuild", &NeurologicalStructureInitializationInfo::optimizerBuild)
        .def_readwrite("clippingMethodBuild", &NeurologicalStructureInitializationInfo::clippingMethodBuild);

    pybind11::class_<derivativesApplyingInfo>(_handle, "derivativesApplyingInfo")
        .def(pybind11::init<>())
        .def_readwrite("learningRate", &derivativesApplyingInfo::learningRate)
        .def_readwrite("averagingDivisor", &derivativesApplyingInfo::averagingDivisor);

    _handle.def("ReadNeurologicalStructure", &ReadNeurologicalStructure);
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
