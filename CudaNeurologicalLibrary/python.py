import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")

from .build.Release import cuda_module as cuModule

class CudaNeurologicalLibrary:
    # ===== Optimizers =====
    @staticmethod
    def OPT_ADAM(beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        opt = cuModule.ADAMOptimizerBuildInfo()
        opt.beta1 = beta1
        opt.beta2 = beta2
        opt.epsilon = epsilon
        return opt
    
    # ===== Activation Functions =====
    @staticmethod
    def AF_ReLU(fade_multiplier: float):
        relu = cuModule.ReLUActivationBuildInfo()
        relu.fadeMultiplier = fade_multiplier
        return relu
    
    # ===== Build Info Constructors =====
    @staticmethod
    def RMSNORM(tensor_size: int, freeze: bool = False, residual: bool = False):
        layer = cuModule.RMSNormLayerBuildInfo()
        layer.tensorSize = tensor_size
        layer.freeze = freeze
        layer.residual = residual
        return layer

    @staticmethod
    def DENSE(input_size: int, output_size: int, activation=None, freeze: bool = False, residual: bool = False, max_dropout = 0):
        dense = cuModule.DenseLayerBuildInfo()
        dense.inputSize = input_size
        dense.outputSize = output_size
        
        dense.maxDropout = max_dropout

        if activation is not None:
            dense.activation = activation

        dense.freeze = freeze
        dense.residual = residual
        return dense

    @staticmethod
    def CONVOLUTIONAL(input_channels: int, output_channels: int,
             slide_width: int, slide_height: int,
             input_width: int, input_height: int,
             stride: int = 1, padding: int = 0,
             activation=None, freeze: bool = False, residual: bool = False, max_dropout = 0):
        conv = cuModule.ConvolutionalLayerBuildInfo()
        conv.outputChannels = output_channels
        conv.inputChannels = input_channels
        conv.slideWidth = slide_width
        conv.slideHeight = slide_height
        conv.inputWidth = input_width
        conv.inputHeight = input_height
        conv.stride = stride
        conv.padding = padding

        conv.maxDropout = max_dropout

        if activation is not None:
            conv.activation = activation

        conv.freeze = freeze
        conv.residual = residual
        return conv

    @staticmethod
    def MH_MASKED_SELF_ATTENTION(input_size: int, head_output_size: int, heads_amount: int, freeze: bool = False, residual: bool = False):
        attn = cuModule.MultiHeadMaskedSelfAttentionLayerBuildInfo()
        attn.inputSize = input_size
        attn.headOutputSize = head_output_size
        attn.headsAmount = heads_amount
        attn.freeze = freeze
        attn.residual = residual
        return attn

    @staticmethod
    def SCC_POSITIONAL_EMBEDDING(input_size: int, output_size: int, freeze: bool = False, residual: bool = False):
        scc = cuModule.SCCPositionalEmbeddingLayerBuildInfo()
        scc.inputSize = input_size
        scc.outputSize = output_size
        scc.freeze = freeze
        scc.residual = residual
        return scc

    @staticmethod
    def CHAIN(components: list, freeze: bool = False, residual: bool = False):
        chain = cuModule.ComponentsChainBuildInfo()
        chain.components = components
        chain.freeze = freeze
        chain.residual = residual
        return chain

    # ===== Control Functions =====

    @staticmethod
    def Build(build_info, optimization = None):
        if optimization == None:
            optimization = cuModule.BaseOptimizerBuildInfo()
        return cuModule.BuildModel(build_info, optimization)

    @staticmethod
    def Initialize(model_id: str, seed: int):
        cuModule.InitializeModel(model_id, seed)

    @staticmethod
    def Infer(model_id: str, input_buffer, output_buffer, sequence_length: int):
        cuModule.InferModel(model_id, input_buffer, output_buffer, sequence_length)
    @staticmethod
    def ClearInferMemory(model_id: str):
        cuModule.ClearModelInference(model_id)

    @staticmethod
    def Activate(model_id: str, input_buffer, output_buffer,
                 batch_size: int, series_lengths, total_samples: int):
        cuModule.ActivateModel(
            model_id,
            input_buffer,
            output_buffer,
            batch_size,
            series_lengths,
            total_samples
        )

    @staticmethod
    def Adjust(model_id: str, output_grad_buffer, input_grad_buffer,
               batch_size: int, series_lengths, total_samples: int):
        cuModule.AdjustModel(
            model_id,
            output_grad_buffer,
            input_grad_buffer,
            batch_size,
            series_lengths,
            total_samples
        )

    @staticmethod
    def Update(model_id: str, learning_alpha: float):
        cuModule.UpdateModel(model_id, learning_alpha)
        
    @staticmethod
    def Extract(model_id: str):
        return cuModule.ExtractModel(model_id)

    @staticmethod
    def Insert(model_id: str, save_info):
        return cuModule.InsertModel(model_id, save_info)

    @staticmethod
    def SaveInfoToDict(save_info) -> dict:
        type_name = save_info.componentType.name

        if type_name == "COMPONENTS_CHAIN":
            return {
                "type": "COMPONENTS_CHAIN",
                "components": [CudaNeurologicalLibrary.SaveInfoToDict(c) for c in save_info.components]
            }
        elif type_name == "DENSE_LAYER":
            return {
                "type": "DENSE_LAYER",
                "weights": list(save_info.weights),
                "biases":  list(save_info.biases)
            }
        elif type_name == "CONVOLUTIONAL_LAYER":
            return {
                "type": "CONVOLUTIONAL_LAYER",
                "weights": list(save_info.weights),
                "biases":  list(save_info.biases)
            }
        elif type_name == "RMSNORM_LAYER":
            return {
                "type": "RMSNORM_LAYER",
                "gamma": list(save_info.gamma)
            }
        elif type_name == "MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER":
            return {
                "type": "MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER",
                "weights": list(save_info.weights)
            }
        elif type_name == "SCC_POSITIONAL_EMBEDDING_LAYER":
            return {
                "type": "SCC_POSITIONAL_EMBEDDING_LAYER",
                "weights": list(save_info.weights)
            }
        else:
            raise ValueError(f"Unknown component type: {type_name}")

    @staticmethod
    def DictToSaveInfo(d: dict):
        type_name = d["type"]

        if type_name == "COMPONENTS_CHAIN":
            save_info = cuModule.ComponentsChainSaveInfo()
            save_info.componentType = cuModule.ComponentType.COMPONENTS_CHAIN

            save_info.components = [CudaNeurologicalLibrary.DictToSaveInfo(c) for c in d["components"]]
            return save_info
        elif type_name == "DENSE_LAYER":
            save_info = cuModule.DenseLayerSaveInfo()
            save_info.componentType = cuModule.ComponentType.DENSE_LAYER

            save_info.weights = d["weights"]
            save_info.biases  = d["biases"]
            return save_info
        elif type_name == "CONVOLUTIONAL_LAYER":
            save_info = cuModule.ConvolutionalLayerSaveInfo()
            save_info.componentType = cuModule.ComponentType.CONVOLUTIONAL_LAYER

            save_info.weights = d["weights"]
            save_info.biases  = d["biases"]
            return save_info
        elif type_name == "RMSNORM_LAYER":
            save_info = cuModule.RMSNormLayerSaveInfo()
            save_info.componentType = cuModule.ComponentType.RMSNORM_LAYER

            save_info.gamma = d["gamma"]
            return save_info
        elif type_name == "MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER":
            save_info = cuModule.MultiHeadMaskedSelfAttentionLayerSaveInfo()
            save_info.componentType = cuModule.ComponentType.MULTI_HEAD_MASKED_SELF_ATTENTION_LAYER

            save_info.weights = d["weights"]
            return save_info
        elif type_name == "SCC_POSITIONAL_EMBEDDING_LAYER":
            save_info = cuModule.SCCPositionalEmbeddingLayerSaveInfo()
            save_info.componentType = cuModule.ComponentType.SCC_POSITIONAL_EMBEDDING_LAYER

            save_info.weights = d["weights"]
            return save_info
        else:
            raise ValueError(f"Unknown component type: {type_name}")

    # LOSS
    @staticmethod
    def MSE(prediction, correction, propagation):
        cuModule.MSE(prediction, correction, propagation)

    # EXTRA
    @staticmethod
    def _softmax(values, eachSize):
        cuModule.softmax(values, eachSize)