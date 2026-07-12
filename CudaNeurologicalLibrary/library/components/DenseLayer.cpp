#include "DenseLayer.h"

#include "../cuda/activations.cuh"
#include "../cuda/adjustments.cuh"
#include "../cuda/initialization.cuh"
#include "../cuda/optimization.cuh"

#include "../activation_functions/ReLU.h"
#include "../optimization/ADAM_optimization.h"

#include <cuda_runtime.h>

// Constructor
cuda_DenseLayer::cuda_DenseLayer(int inSize, int outSize, int _maxDropout)
    : inputSize(inSize),
      outputSize(outSize),
      weights(inSize * outSize),
      biases(outSize),
      weights_gradient(inSize * outSize),
      biases_gradient(outSize),

      maxDropout(_maxDropout),
      dropout_mask(_maxDropout)
{
    // 1. Obtain a random seed from hardware for drop out masking
    std::random_device rd;

    // 2. Initialize the Mersenne Twister engine with the seed for drop out masking
    std::mt19937 gen(rd());

    // 3. Define the range [min, max] (both inclusive)
    std::uniform_int_distribution<int> distrib(0, outputSize-1);

    this->dropout_rand = gen;
    this->dropout_rand_distribution = distrib;
}

// Size accessors
int cuda_DenseLayer::in_size() const {
    return inputSize;
}

int cuda_DenseLayer::out_size() const {
    return outputSize;
}

// Static: Initialize
void cuda_DenseLayer::Initialize(cuda_DenseLayer& layer, unsigned long seed) {
    fc_weights__initialization(
        layer.inputSize,
        layer.outputSize,
        layer.weights.data(),
        seed
    );
}

// Static: Forward Pass
void cuda_DenseLayer::ForwardPass(
    cuda_DenseLayer& layer,
    cuda_ConstantTensorSpan input,
    cuda_TensorSpan output,
    const int& samplesAmount
) {
    layer.cached_input = input;

    // produce dropout mask
    int dropoutSize = layer.maxDropout;
    int i = 0;
    for (int _ = 0; _ < layer.dropout_mask.size; ++_) {
        const int rand_activation = layer.dropout_rand_distribution(layer.dropout_rand);
        bool itContains = false;
        for (int j = 0; j < layer.maxDropout; ++j) {
            if (layer.dropout_mask.data()[j] == rand_activation) {
                itContains = true;
            }
        }
        if (itContains) {
            dropoutSize -= 1;
            continue;
        }

        // printf("random activation: %i; i: %i\n", rand_activation, i);
        layer.dropout_mask.data()[i] = rand_activation;
        // printf("i: %i; activation: %i; size: %i\n", i, rand_activation, layer.dropout_mask.size);
        i++;
    }

    const cuda_neurologicalValue activationMultiplier = 1.0f/(1.0f-(dropoutSize/layer.outputSize));

    fc_weights__activation(
        input.data(),
        output.data(),
        layer.weights.data(),
        samplesAmount,
        layer.inputSize,
        layer.outputSize
    );
    cudaDeviceSynchronize();
    
    if (layer.activation != nullptr) {
        switch(layer.activation->activationType) {
            case cuda_ActivationFunctionTypes::ReLU: {
                cuda_ReLU* relu = static_cast<cuda_ReLU*>(layer.activation.get());
                if (relu->cache_output.size < output.size) {
                    relu->cache_output = cuda_Tensor(output.size);
                }
                relu_af__activation(
                    output.data(),
                    layer.biases.data(),
                    samplesAmount,
                    layer.outputSize,

                    relu->cache_output.data(),
                    relu->fadeMultiplier,
            
                    activationMultiplier
                );
                break;
            }
            default: {
                linear_af__activation(
                    output.data(),
                    layer.biases.data(),
                    samplesAmount,
                    layer.outputSize,
            
                    activationMultiplier
                );
                break;
            }
        }
    } else {
        linear_af__activation(
            output.data(),
            layer.biases.data(),
            samplesAmount,
            layer.outputSize,

            activationMultiplier
        );
    }
    cudaDeviceSynchronize();

    if (dropoutSize > 0) {
        // printf("dropout: %i; mask size: %i\n", dropoutSize, layer.dropout_mask.size);
        dropout(output.data(), layer.dropout_mask.data(), layer.out_size(), dropoutSize, samplesAmount);
        cudaDeviceSynchronize();
    }
    layer.droppedOut = dropoutSize;
}

// Static: Infer
void cuda_DenseLayer::Infer(
    cuda_DenseLayer& layer,
    cuda_ConstantTensorSpan input,
    cuda_TensorSpan output,
    const int& sequenceLength
) {
    fc_weights__activation(
        input.data(),
        output.data(),
        layer.weights.data(),
        sequenceLength,
        layer.inputSize,
        layer.outputSize
    );
    cudaDeviceSynchronize();

    if (layer.activation != nullptr) {
        switch(layer.activation->activationType) {
            case cuda_ActivationFunctionTypes::ReLU: {
                cuda_ReLU* relu = static_cast<cuda_ReLU*>(layer.activation.get());
                if (relu->cache_output.size < output.size) {
                    relu->cache_output = cuda_Tensor(output.size);
                }
                relu_af__activation(
                    output.data(),
                    layer.biases.data(),
                    sequenceLength,
                    layer.outputSize,
                    relu->cache_output.data(),
                    relu->fadeMultiplier,

                    1
                );
                break;
            }
            default: {
                linear_af__activation(
                    output.data(),
                    layer.biases.data(),
                    sequenceLength,
                    layer.outputSize,

                    1
                );
                break;
            }
        }
    } else {
        linear_af__activation(
            output.data(),
            layer.biases.data(),
            sequenceLength,
            layer.outputSize,

            1
        );
    }
    cudaDeviceSynchronize();
}
void cuda_DenseLayer::ClearInference(cuda_DenseLayer& layer) {
    // EMPTY
}

// Static: Backpropagation
void cuda_DenseLayer::BackPropagation(
    cuda_DenseLayer& layer,
    cuda_ConstantTensorSpan output_gradient,
    cuda_TensorSpan input_gradient,
    const int& samplesAmount
) {
    // linear_dense__adjustment(
    //     output_gradient.data(),
    //     layer.cached_input.data(),
    //     input_gradient.data(),
    //     layer.weights.data(),
    //     layer.weights_gradient.data(),
    //     layer.biases_gradient.data(),
    //     samplesAmount,
    //     layer.inputSize,
    //     layer.outputSize
    // );
    
    const cuda_neurologicalValue* gradient = output_gradient.data();

    const int dropoutSize = layer.droppedOut;
    if (dropoutSize > 0) {
        if (layer.dropout_gradient.size < samplesAmount*layer.out_size()) {
            layer.dropout_gradient = cuda_Tensor(samplesAmount*layer.out_size());
        }
        layer.dropout_gradient.zeroAll();
        layer.dropout_gradient.copyExactData(output_gradient.data());
        dropout(layer.dropout_gradient.data(), layer.dropout_mask.data(), layer.out_size(), dropoutSize, samplesAmount);
        cudaDeviceSynchronize();
        gradient = layer.dropout_gradient.data();
    }

    if (layer.activation != nullptr) {
        switch(layer.activation->activationType) {
            case cuda_ActivationFunctionTypes::ReLU: {
                cuda_ReLU* relu = static_cast<cuda_ReLU*>(layer.activation.get());
                if (relu->activation_gradient.size < output_gradient.size) {
                    relu->activation_gradient = cuda_Tensor(output_gradient.size);
                }
                relu_af__gradient(
                    output_gradient.data(),
                    relu->activation_gradient.data(),

                    layer.biases_gradient.data(),
                    samplesAmount,
                    layer.outputSize,

                    relu->cache_output.data(),
                    relu->fadeMultiplier
                );
                cudaDeviceSynchronize();
                gradient = relu->activation_gradient.data();
                break;
            }
            default: {
                linear_af__gradient(
                    output_gradient.data(),
                    layer.biases_gradient.data(),
                    samplesAmount,
                    layer.outputSize
                );
                cudaDeviceSynchronize();
                break;
            }
        }
    } else {
        linear_af__gradient(
            output_gradient.data(),
            layer.biases_gradient.data(),
            samplesAmount,
            layer.outputSize
        );
        cudaDeviceSynchronize();
    }
    fc_weights__gradient(
        gradient,
        layer.cached_input.data(),
        layer.weights_gradient.data(),
        samplesAmount,
        layer.inputSize,
        layer.outputSize
    );
    fc_input__gradient(
        gradient,
        layer.weights.data(),
        input_gradient.data(),
        samplesAmount,
        layer.inputSize,
        layer.outputSize
    );
    cudaDeviceSynchronize();
}

// Static: Apply parameter update
void cuda_DenseLayer::Update(cuda_DenseLayer& layer, const cuda_neurologicalValue& updateMultiplier) {
    if (layer.biases_optimization != nullptr) {
        switch(layer.biases_optimization->optimizerType) {
            case cuda_OptimizerTypes::ADAM: {
                cuda_ADAM_optimization* adam = static_cast<cuda_ADAM_optimization*>(layer.biases_optimization.get());
                ADAM_optimization__apply(
                    layer.out_size(),
                    adam->beta1,
                    adam->beta2,
                    adam->epsilon,
                    adam->t,
                    adam->m.data(),
                    adam->v.data(),
                    layer.biases.data(),
                    layer.biases_gradient.data(),
                    updateMultiplier
                );
                break;
            }
            default: {
                default__apply(
                    layer.out_size(),
                    layer.biases.data(),
                    layer.biases_gradient.data(),
                    updateMultiplier
                );
                break;
            }
        }
    } else {
        default__apply(
            layer.out_size(),
            layer.biases.data(),
            layer.biases_gradient.data(),
            updateMultiplier
        );
    }
    if (layer.weights_optimization != nullptr) {
        switch(layer.weights_optimization->optimizerType) {
            case cuda_OptimizerTypes::ADAM: {
                cuda_ADAM_optimization* adam = static_cast<cuda_ADAM_optimization*>(layer.weights_optimization.get());
                ADAM_optimization__apply(
                    layer.in_size()*layer.out_size(),
                    adam->beta1,
                    adam->beta2,
                    adam->epsilon,
                    adam->t,
                    adam->m.data(),
                    adam->v.data(),
                    layer.weights.data(),
                    layer.weights_gradient.data(),
                    updateMultiplier
                );
                break;
            }
            default: {
                default__apply(
                    layer.in_size()*layer.out_size(),
                    layer.weights.data(),
                    layer.weights_gradient.data(),
                    updateMultiplier
                );
                break;
            }
        }
    } else {
        default__apply(
            layer.in_size()*layer.out_size(),
            layer.weights.data(),
            layer.weights_gradient.data(),
            updateMultiplier
        );
    }
    cudaDeviceSynchronize();
    layer.weights_gradient.zeroAll();
    layer.biases_gradient.zeroAll();
}