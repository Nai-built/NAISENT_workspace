from functools import singledispatchmethod
from NeurologicalLibrary.bridge.Pybind11ToJson import pybind_to_dict
import NeurologicalLibrary.bridge.build.Release.simple_module as cppModule

import json

def multiply(n1, n2):
    return cppModule.multiply(n1, n2)
def add(n1, n2):
    return cppModule.add(n1, n2)
def nextInteger():
    return cppModule.nextInteger()

def TurnParameterDataToSaveInfo(parameterData):
    saveInfo = cppModule.NeurologicalParameterSaveInfo()
    saveInfo.value = parameterData['value']

    if 'optimizationInfo' in parameterData:
        optimizationSaveInfo = cppModule.NeurologicalParameterOptimizationSaveInfo()
        optimizationSaveInfo.optimizerType = parameterData['optimizationInfo']['optimizerType']

        if parameterData['optimizationInfo']['optimizerType'] == "ADAM":
            optimizationSaveInfo.v = parameterData['optimizationInfo']['v']
            optimizationSaveInfo.m = parameterData['optimizationInfo']['m']
            optimizationSaveInfo.t = parameterData['optimizationInfo']['t']
        
        saveInfo.optimizationInfo = optimizationSaveInfo

    return saveInfo

def TurnDenseNeuronDataToSaveInfo(denseNeuronData):
    saveInfo = cppModule.DenseNeuronSaveInfo()
    saveInfo.bias = TurnParameterDataToSaveInfo(denseNeuronData['bias'])

    weights = []
    for i in range(len(denseNeuronData['weights'])):
        weights.insert(i, TurnParameterDataToSaveInfo(denseNeuronData['weights'][i]))
    saveInfo.weights = weights

    return saveInfo
def TurnConvolutionalFilterDataToSaveInfo(convolutionalFilterData):
    saveInfo = cppModule.ConvolutionalFilterSaveInfo()
    saveInfo.bias = TurnParameterDataToSaveInfo(convolutionalFilterData['bias'])

    kernels = []
    for kI in range(len(convolutionalFilterData['kernels'])):
        kernel = []
        kernelSaveInfo = convolutionalFilterData['kernels'][kI]

        for _y in range(len(kernelSaveInfo)):
            kernelRow = []
            kernelRowSaveInfo = kernelSaveInfo[_y]
            for _x in range(len(kernelRowSaveInfo)):
                kernelRow.insert(_x, TurnParameterDataToSaveInfo(kernelRowSaveInfo[_x]))
                pass
            kernel.insert(_y, kernelRow)
            pass
        kernels.insert(kI, kernel)
        pass
    saveInfo.kernels = kernels

    return saveInfo

def TurnDataToNeurologicalSaveInfo(data):
    saveInfo = cppModule.NeurologicalComponentSaveInfo()
    if not 'componentType' in data:
        saveInfo.componentType = "NONE"
        return saveInfo
    print("READING:", data['componentType'])
    saveInfo.componentType = data['componentType']

    if data['componentType'] == "DENSE-LAYER":
        _denseNeurons = []
        for i in range(len(data['denseNeurons'])):
            # print("dense neuron", i)
            _denseNeurons.insert(i, TurnDenseNeuronDataToSaveInfo(data['denseNeurons'][i]))

        saveInfo.denseNeurons = _denseNeurons
    elif data['componentType'] == "RECURSIVE.LSTM-LAYER":
        saveInfo.forgetGateNeurons = TurnDataToNeurologicalSaveInfo\
        ({
            'componentType': "DENSE-LAYER",
            'denseNeurons': data['forgetGateNeurons'],
        }).denseNeurons
        saveInfo.inputGateNeurons = TurnDataToNeurologicalSaveInfo\
        ({
            'componentType': "DENSE-LAYER",
            'denseNeurons': data['inputGateNeurons'],
        }).denseNeurons
        saveInfo.candidateMemoryGateNeurons = TurnDataToNeurologicalSaveInfo\
        ({
            'componentType': "DENSE-LAYER",
            'denseNeurons': data['candidateMemoryGateNeurons'],
        }).denseNeurons
        saveInfo.outputGateNeurons = TurnDataToNeurologicalSaveInfo\
        ({
            'componentType': "DENSE-LAYER",
            'denseNeurons': data['outputGateNeurons'],
        }).denseNeurons
    elif data['componentType'] == "CONVOLUTIONAL-LAYER":
        _convolutionalFilters = []
        for i in range(len(data['convolutionalFilters'])):
            # print("convolutional filter", i)
            _convolutionalFilters.insert(i, TurnConvolutionalFilterDataToSaveInfo(data['convolutionalFilters'][i]))

        saveInfo.convolutionalFilters = _convolutionalFilters
    elif data['componentType'] == "CHAIN":
        _chain = []
        for i in range(len(data['chain'])):
            _chain.insert(i, TurnDataToNeurologicalSaveInfo(data['chain'][i]))
        
        saveInfo.chain = _chain

    return saveInfo

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
    @staticmethod
    def Read(structureId):
        data = pybind_to_dict(cppModule.ReadNeurologicalStructure(structureId))
        return data
    @staticmethod
    def Write(structureId, data):
        cppModule.WriteNeurologicalStructure(structureId, TurnDataToNeurologicalSaveInfo(data))
    
    # OPTIMIZERS
    @staticmethod
    def OPTIMIZER_ADAM(epsilon, beta1, beta2):
        buildInfo = cppModule.OptimizerBuildInfo()
        buildInfo.optimizerType = "ADAM"

        buildInfo.epsilon = epsilon
        buildInfo.beta1 = beta1
        buildInfo.beta2 = beta2

        return buildInfo

    # CLIPPING METHODS
    @staticmethod
    def CLIPPING_METHOD__NORM(clip_value):
        buildInfo = cppModule.ClippingMethodBuildinfo()
        buildInfo.optimizerType = "NORM"

        buildInfo.clip_value = clip_value

        return buildInfo

    # COMPONENTS
    @staticmethod
    def DENSE(inputSize, outputSize):
        buildInfo = cppModule.NeurologicalComponentBuildInfo()
        buildInfo.componentType = "DENSE-LAYER"
        buildInfo.inputSize = inputSize
        buildInfo.outputSize = outputSize

        return buildInfo

    @staticmethod
    def LSTM_RECURSIVE(inputSize, outputSize):
        buildInfo = cppModule.NeurologicalComponentBuildInfo()
        buildInfo.componentType = "RECURSIVE.LSTM-LAYER"
        buildInfo.inputSize = inputSize
        buildInfo.outputSize = outputSize

        return buildInfo
    
    @staticmethod
    def CONVOLUTIONAL(inputSize, outputSize, padding, stride, kernelSize):
        buildInfo = cppModule.NeurologicalComponentBuildInfo()
        buildInfo.componentType = "CONVOLUTIONAL-LAYER"
        buildInfo.inputSize = inputSize
        buildInfo.outputSize = outputSize

        buildInfo.padding = padding
        buildInfo.stride = stride
        buildInfo.kernelSize = kernelSize

        return buildInfo
    @staticmethod
    def AVERAGE_POOLING(stride, poolSize):
        buildInfo = cppModule.NeurologicalComponentBuildInfo()
        buildInfo.componentType = "CONVOLUTIONAL-POOLER"
        
        buildInfo.poolingType = "AVERAGE"
        
        buildInfo.stride = stride
        buildInfo.poolSize = poolSize

        return buildInfo
    @staticmethod
    def MAX_POOLING(stride, poolSize):
        buildInfo = cppModule.NeurologicalComponentBuildInfo()
        buildInfo.componentType = "CONVOLUTIONAL-POOLER"

        buildInfo.poolingType = "MAX"
        
        buildInfo.stride = stride
        buildInfo.poolSize = poolSize

        return buildInfo
    @staticmethod
    def MIN_POOLING(stride, poolSize):
        buildInfo = cppModule.NeurologicalComponentBuildInfo()
        buildInfo.componentType = "CONVOLUTIONAL-POOLER"
        
        buildInfo.poolingType = "MIN"
        
        buildInfo.stride = stride
        buildInfo.poolSize = poolSize

        return buildInfo
    @staticmethod
    def FLATTENER():
        buildInfo = cppModule.NeurologicalComponentBuildInfo()
        buildInfo.componentType = "CHANNELS-FLATTENER"

        return buildInfo
    @staticmethod
    def SPREADER(amount, width, height):
        buildInfo = cppModule.NeurologicalComponentBuildInfo()
        buildInfo.componentType = "CHANNELS-SPREADER"

        buildInfo.expectedChannelsAmount = amount
        buildInfo.expectedChannelsWidth = width
        buildInfo.expectedChannelsHeight = height

        return buildInfo
    
    @staticmethod
    def CHAIN(_chain):
        buildInfo = cppModule.NeurologicalComponentBuildInfo()
        buildInfo.componentType = "CHAIN"
        buildInfo.chain = _chain

        return buildInfo
    
    @staticmethod
    def AF_ReLU(_leakage):
        buildInfo = cppModule.NeurologicalComponentBuildInfo()
        buildInfo.componentType = "AF-ReLU"
        buildInfo._negativeLeakage = _leakage

        return buildInfo
    
    @staticmethod
    def AF_Tanh():
        buildInfo = cppModule.NeurologicalComponentBuildInfo()
        buildInfo.componentType = "AF-Tanh"

        return buildInfo
    
    @staticmethod
    def AF_Softmax(ignoreJacobian = False):
        buildInfo = cppModule.NeurologicalComponentBuildInfo()
        buildInfo.componentType = "AF-Softmax"
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