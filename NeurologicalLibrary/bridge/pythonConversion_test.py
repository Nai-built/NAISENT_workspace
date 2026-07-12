from functools import singledispatchmethod
from bridge.Pybind11ToJson import pybind_to_dict
import bridge.build.Release.simple_module as cppModule

import json

def multiply(n1, n2):
    return cppModule.multiply(n1, n2)
def add(n1, n2):
    return cppModule.add(n1, n2)
def nextInteger():
    return cppModule.nextInteger()

class NeurologicalLogic:
    # CREATION
    @staticmethod
    def CreateNeurologicalStructure(structureBuild, optimizer = None, clippingMethod = None):
        structureId = cppModule.CreateNeurologicalStructure(structureBuild)

        initializationInfo = cppModule.NeurologicalStructureInitializationInfo()
        initializationInfo.optimizerBuild = optimizer
        initializationInfo.clippingMethodBuild = clippingMethod
        cppModule.InitializeNeurologicalStructure(structureId, initializationInfo)

        return structureId
    
    # USAGE
    @staticmethod
    def Activate(structureId, input, sampleIndex):
        return cppModule.ActivateNeurologicalStructure(structureId, input, sampleIndex)
    @staticmethod
    def Adjust(structureId, gradient, sampleIndex):
        return cppModule.AdjustNeurologicalStructure(structureId, gradient, sampleIndex)
    @staticmethod
    def ApplyAdjustments(structureId, learningRate, averagingDivisor):
        applyingInfo = cppModule.derivativesApplyingInfo()
        applyingInfo.learningRate = learningRate
        applyingInfo.averagingDivisor = averagingDivisor

        return cppModule.ApplyNeurologicalStructureAdjustments(structureId, applyingInfo)
    def Read(structureId):
        data = pybind_to_dict(cppModule.ReadNeurologicalStructure(structureId))
        return json.dumps(data, indent=2)
    
    # OPTIMIZERS
    @staticmethod
    def OPTIMIZER_ADAM(epsilon, beta1, beta2):
        buildInfo = cppModule.AdamOptimizerBuildInfo()
        buildInfo.epsilon = epsilon
        buildInfo.beta1 = beta1
        buildInfo.beta2 = beta2

        return buildInfo

    # CLIPPING METHODS
    @staticmethod
    def CLIPPING_METHOD__NORM(clip_value):
        buildInfo = cppModule.NormClippingMethodBuildInfo()
        buildInfo.clip_value = clip_value

        return buildInfo

    # COMPONENTS
    @staticmethod
    def DENSE(inputSize, outputSize):
        buildInfo = cppModule.DenseLayerBuildInfo()
        buildInfo.inputSize = inputSize
        buildInfo.outputSize = outputSize

        return buildInfo

    @staticmethod
    def CHAIN(_chain):
        buildInfo = cppModule.ChainBuildInfo()
        buildInfo.chain = _chain

        return buildInfo
    
    @staticmethod
    def AF_ReLU(_leakage):
        buildInfo = cppModule.AF_ReLUBuildInfo()
        buildInfo._negativeLeakage = _leakage

        return buildInfo
    
    @staticmethod
    def AF_Tanh():
        buildInfo = cppModule.AF_TanhBuildInfo()

        return buildInfo
    
    @staticmethod
    def AF_Softmax(ignoreJacobian = False):
        buildInfo = cppModule.AF_SoftmaxBuildInfo()
        buildInfo._ignoreJacobian = ignoreJacobian

        return buildInfo
    
    # LOSS FUNCTIONS
    @staticmethod
    def ZERO_GRADIENT(size: int):
        return cppModule.zero_loss_gradient(size)
    
    @staticmethod
    def MSE_LOSS_GRADIENT(prediction: list, desiredOutput: list):
        # gradient = [None] * len(result)
        # for i in range(len(result)):
        #    gradient[i] = result[i] - desiredResult[i]

        # return gradient 

        return cppModule.MSE_loss_gradient(prediction, desiredOutput)
    @staticmethod
    def MSE_LOSS_VALUE(size: int):
        return cppModule.MSE_loss_value(size)
    
    @staticmethod
    def KL_DIVERGENCE_GRADIENT(prediction: list, desiredOutput: list):
        return cppModule.KL_divergence_gradient(prediction, desiredOutput)
    @staticmethod
    def KL_DIVERGENCE_VALUE(size: int):
        return cppModule.KL_divergence_value(size)