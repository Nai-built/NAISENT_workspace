import OptimizedNeurologicalLibrary.build.Release.simple_module as cppModule

from array import array

def SaveInfoToData(saveInfo):
    if saveInfo.saveType == "chain":
        saveDictionary = {}
        saveDictionary['saveType'] = "chain"
        _chain = [None] * len(saveInfo.componentsChainSave)
        for i, e in enumerate(saveInfo.componentsChainSave):
            _chain[i] = SaveInfoToData(e)
        saveDictionary["componentsChainSave"] = _chain

        return saveDictionary
    elif saveInfo.saveType == "common-layer":
        saveDictionary = {'saveType': "common-layer", 'weights': saveInfo.weights, "biases": saveInfo.biases}

        return saveDictionary
    elif saveInfo.saveType == "transformer":
        saveDictionary = {'saveType': "transformer",
                          'encodeWeights': saveInfo.encodeWeights,
                          'attentionWeights': saveInfo.attentionWeights, "attentionFinalBiases": saveInfo.attentionFinalBiases,
                          'normalizationGammas': saveInfo.normalizationGammas,
                          'feedForwardNetworksWeights': saveInfo.feedForwardNetworksWeights, "feedForwardNetworksBiases": saveInfo.feedForwardNetworksBiases,
                          'decodeWeights': saveInfo.decodeWeights, "decodeBiases": saveInfo.decodeBiases}

        return saveDictionary

def DataToSaveInfo(saveDictionary):
    if saveDictionary['saveType'] == "chain":
        saveInfo = cppModule.ComponentsChainSaveInfo()
        saveInfo.saveType = "chain"
        _chain = [None] * len(saveDictionary['componentsChainSave'])
        for i, e in enumerate(saveDictionary['componentsChainSave']):
            _chain[i] = DataToSaveInfo(e)
        saveInfo.componentsChainSave = _chain

        return saveInfo
    elif saveDictionary['saveType'] == "common-layer":
        saveInfo = cppModule.CommonLayerSaveInfo()
        saveInfo.saveType = "common-layer"

        saveInfo.weights = saveDictionary['weights']
        saveInfo.biases = saveDictionary['biases']

        return saveInfo
    elif saveDictionary['saveType'] == "transformer":
        saveInfo = cppModule.TransformerSaveInfo()
        saveInfo.saveType = "transformer"

        saveInfo.encodeWeights = saveDictionary['encodeWeights']

        saveInfo.attentionWeights = saveDictionary['attentionWeights']
        saveInfo.attentionFinalBiases = saveDictionary['attentionFinalBiases']
        
        saveInfo.normalizationGammas = saveDictionary['normalizationGammas']
        
        saveInfo.feedForwardNetworksWeights = saveDictionary['feedForwardNetworksWeights']
        saveInfo.feedForwardNetworksBiases = saveDictionary['feedForwardNetworksBiases']
        
        saveInfo.decodeWeights = saveDictionary['decodeWeights']
        saveInfo.decodeBiases = saveDictionary['decodeBiases']

        return saveInfo

class OptimizedNeurologicalLibrary:
    # CREATION
    def Build(buildInfo):
        structureId = cppModule.BuildNeurologicalStructure(buildInfo)

        cppModule.InitializeNeurologicalStructure(structureId)

        return structureId
    
    # ACTIVATION FUNCTIONS
    def ReLU(_leakage = 0):
        buildInfo = cppModule.ReLU__BuildInfo()
        buildInfo.activationType = cppModule.ActivationFunctionTypes.RELU
        buildInfo.leakage = _leakage

        return buildInfo

    # CHANNELS POOLERS
    def MAX_POOL(poolingSize: str, stride: int):
        buildInfo = cppModule.MaxPool__BuildInfo()
        buildInfo.poolerType = cppModule.ChannelsPoolerTypes.MAX
        
        buildInfo.poolingSize = poolingSize
        buildInfo.stride = stride

        return buildInfo
        
    def AVERAGE_POOL(poolingSize: str, stride: int):
        buildInfo = cppModule.AveragePool__BuildInfo()
        buildInfo.poolerType = cppModule.ChannelsPoolerTypes.AVERAGE
        
        buildInfo.poolingSize = poolingSize
        buildInfo.stride = stride

        return buildInfo
        
    def MIN_POOL(poolingSize: str, stride: int):
        buildInfo = cppModule.MinPool__BuildInfo()
        buildInfo.poolerType = cppModule.ChannelsPoolerTypes.MIN
        
        buildInfo.poolingSize = poolingSize
        buildInfo.stride = stride

        return buildInfo

    # COMPONENTS
    def CHAIN(_chain):
        buildInfo = cppModule.ComponentsChainBuildInfo()
        buildInfo.componentType = cppModule.NeurologicalComponentTypes.COMPONENTS_CHAIN
        buildInfo.componentsChainBuild = _chain

        return buildInfo
    
    def DENSE(inputSize, outputSize, activation = None):
        buildInfo = cppModule.DenseLayerBuildInfo()
        buildInfo.componentType = cppModule.NeurologicalComponentTypes.DENSE_LAYER
        buildInfo.inputSize = inputSize
        buildInfo.outputSize = outputSize

        buildInfo.activationFunction = activation

        return buildInfo
    
    def CONVOLUTIONAL(inputChannels: int, outputChannels: int
                      , stride: int, padding: int
                      , kernelSize: str, channelSize: str
                      , activation = None
                      , pooling = None):
        buildInfo = cppModule.ConvolutionalLayerBuildInfo()
        buildInfo.componentType = cppModule.NeurologicalComponentTypes.CONVOLUTIONAL_LAYER

        buildInfo.inputChannels = inputChannels
        buildInfo.outputChannels = outputChannels
        
        buildInfo.stride = stride
        buildInfo.padding = padding
        
        buildInfo.kernelSize = kernelSize
        buildInfo.inputChannelSize = channelSize

        buildInfo.activationFunction = activation
        buildInfo.channelsPooler = pooling

        return buildInfo
    
    def LSTM_RECURSIVE(inputSize, outputSize, castBeyond = False):
        buildInfo = cppModule.LSTM_RecursiveLayerBuildInfo()
        buildInfo.componentType = cppModule.NeurologicalComponentTypes.RECURSIVE_LSTM
        buildInfo.inputSize = inputSize
        buildInfo.outputSize = outputSize

        buildInfo.castBeyond = castBeyond

        return buildInfo
    
    # TRANSFORMERS
    def FFL(hiddenSize, activation = None):
        buildInfo = cppModule.HiddenFFL_BuildInfo()
        buildInfo.hidden_size = hiddenSize

        buildInfo.activationFunction = activation

        return buildInfo

    def TRANSFORMER_STACK(attentionHeads, ffn):
        buildInfo = cppModule.TransformerStack_BuildInfo()
        buildInfo.attention_heads = attentionHeads
        buildInfo.ffn = ffn

        return buildInfo

    def DECODER_ONLY_TRANSFORMER(inputTokens, outputTokens, encodingSize, stacksBuild):
        buildInfo = cppModule.DecoderOnly_TransformerBuildInfo()
        buildInfo.componentType = cppModule.NeurologicalComponentTypes.DECODER_ONLY_TRANSFORMER
        buildInfo.inputTokens = inputTokens
        buildInfo.outputTokens = outputTokens
        buildInfo.encodingSize = encodingSize

        buildInfo.stacksBuild = stacksBuild

        return buildInfo

    # USAGE
    def Activate(id, input, output, batchSize, seriesLengths = array("i"), totalSamples = 0):
        cppModule.ActivateNeurologicalStructure(id, input, output, batchSize, seriesLengths, totalSamples)
    def Adjust(id, propagation, inputPropagation, lr, batchSize, seriesLengths = array("i"), totalSamples = 0):
        cppModule.AdjustNeurologicalStructure(id, propagation, inputPropagation, lr, batchSize, seriesLengths, totalSamples)

    # LOSS
    def MSE(prediction, correction, propagation):
        cppModule.MSE(prediction, correction, propagation)

    # EXTRA
    def _softmax(values, eachSize):
        cppModule.softmax(values, eachSize)

    # SAVE / LOAD
    def ReadStructureInfo(id):
        return SaveInfoToData(cppModule.ReadNeurologicalStructure(id))
    def WriteStructureInfo(id, saveData):
        cppModule.WriteNeurologicalStructure(id, DataToSaveInfo(saveData))