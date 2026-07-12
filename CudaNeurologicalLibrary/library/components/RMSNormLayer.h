#pragma once

#include "../activation_functions/IActivationFunction.h"
#include "../optimization/IOptimization.h"

#include <memory>

#include "INeurologicalComponent.h"

struct cuda_RMSNormLayer : public cuda_INeurologicalComponent {
private:
    int tensorSize;

public:
    // Parameters Optimization
    std::unique_ptr<cuda_IOptimization> gamma_optimization;

    // Parameters
    cuda_Tensor gamma;   // size = tensorSize

    // Gradients
    cuda_Tensor gamma_gradient;   // size = tensorSize

    // Cache
    cuda_ConstantTensorSpan cached_input;  // a span to the previously passed input
    cuda_Tensor cached_rMultipliers;  // a cache buffer for r multipliers

    // Ctor / Dtor
    cuda_RMSNormLayer(int size);

    // Size accessors
    int size() const;

    // Logic
    // Static operations
    static void Initialize(cuda_RMSNormLayer& layer, unsigned long seed);

    static void ForwardPass(
        cuda_RMSNormLayer& layer,
        cuda_ConstantTensorSpan input,
        cuda_TensorSpan output,
        const int& samplesAmount
    );

    static void Infer(
        cuda_RMSNormLayer& layer,
        cuda_ConstantTensorSpan input,
        cuda_TensorSpan output,
        const int& sequenceLength
    );
    static void ClearInference(
        cuda_RMSNormLayer& layer
    );

    static void BackPropagation(
        cuda_RMSNormLayer& layer,
        cuda_ConstantTensorSpan output_gradient,
        cuda_TensorSpan input_gradient,
        const int& samplesAmount
    );
    
    static void Update(cuda_RMSNormLayer& layer, const cuda_neurologicalValue& updateMultiplier);
};