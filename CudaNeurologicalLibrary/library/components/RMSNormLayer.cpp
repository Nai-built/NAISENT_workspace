#include "RMSNormLayer.h"

#include "../cuda/activations.cuh"
#include "../cuda/adjustments.cuh"
#include "../cuda/initialization.cuh"
#include "../cuda/optimization.cuh"

#include "../optimization/ADAM_optimization.h"

#include <cuda_runtime.h>

// Constructor
cuda_RMSNormLayer::cuda_RMSNormLayer(int size)
    : tensorSize(size),
      gamma(size),
      gamma_gradient(size)
{
}

// Size accessors
int cuda_RMSNormLayer::size() const {
    return tensorSize;
}

// Static: Initialize
void cuda_RMSNormLayer::Initialize(cuda_RMSNormLayer& layer, unsigned long seed) {
    layer.gamma.zeroAll();
    rms__initialization(
        layer.tensorSize,
        layer.gamma.data(),
        seed // not really relevent tbh
    );
}

// Static: Forward Pass
void cuda_RMSNormLayer::ForwardPass(
    cuda_RMSNormLayer& layer,
    cuda_ConstantTensorSpan input,
    cuda_TensorSpan output,
    const int& samplesAmount
) {
    layer.cached_input = input;
    if (layer.cached_rMultipliers.size < samplesAmount) {
        layer.cached_rMultipliers = cuda_Tensor(samplesAmount);
    }
    norm_RMS__activation(
        input.data(),
        output.data(),
        layer.cached_rMultipliers.data(),
        layer.gamma.data(),
        samplesAmount,
        layer.tensorSize
    );
    cudaDeviceSynchronize();
}

// Static: Infer
void cuda_RMSNormLayer::Infer(
    cuda_RMSNormLayer& layer,
    cuda_ConstantTensorSpan input,
    cuda_TensorSpan output,
    const int& sequenceLength
) {
    if (layer.cached_rMultipliers.size < sequenceLength) {
        layer.cached_rMultipliers = cuda_Tensor(sequenceLength);
    }
    norm_RMS__activation(
        input.data(),
        output.data(),
        layer.cached_rMultipliers.data(),
        layer.gamma.data(),
        sequenceLength,
        layer.tensorSize
    );
    cudaDeviceSynchronize();
}
void cuda_RMSNormLayer::ClearInference(cuda_RMSNormLayer& layer) {
    // EMPTY
}

// Static: Backpropagation
void cuda_RMSNormLayer::BackPropagation(
    cuda_RMSNormLayer& layer,
    cuda_ConstantTensorSpan output_gradient,
    cuda_TensorSpan input_gradient,
    const int& samplesAmount
) {
    gamma_RMS__gradient(
        output_gradient.data(),
        layer.cached_input.data(),
        layer.cached_rMultipliers.data(),
        layer.gamma_gradient.data(),
        samplesAmount,
        layer.tensorSize
    );
    input_RMS__gradient(
        output_gradient.data(),
        layer.cached_input.data(),
        layer.cached_rMultipliers.data(),
        layer.gamma.data(),
        input_gradient.data(),
        samplesAmount,
        layer.tensorSize
    );
    cudaDeviceSynchronize();
}

// Static: Apply parameter update
void cuda_RMSNormLayer::Update(cuda_RMSNormLayer& layer, const cuda_neurologicalValue& updateMultiplier) {
    if (layer.gamma_optimization != nullptr) {
        switch(layer.gamma_optimization->optimizerType) {
            case cuda_OptimizerTypes::ADAM: {
                cuda_ADAM_optimization* adam = static_cast<cuda_ADAM_optimization*>(layer.gamma_optimization.get());
                ADAM_optimization__apply(
                    layer.size(),
                    adam->beta1,
                    adam->beta2,
                    adam->epsilon,
                    adam->t,
                    adam->m.data(),
                    adam->v.data(),
                    layer.gamma.data(),
                    layer.gamma_gradient.data(),
                    updateMultiplier
                );
                break;
            }
            default: {
                default__apply(
                    layer.size(),
                    layer.gamma.data(),
                    layer.gamma_gradient.data(),
                    updateMultiplier
                );
                break;
            }
        }
    } else {
        default__apply(
            layer.size(),
            layer.gamma.data(),
            layer.gamma_gradient.data(),
            updateMultiplier
        );
    }
    cudaDeviceSynchronize();
    layer.gamma_gradient.zeroAll();
}