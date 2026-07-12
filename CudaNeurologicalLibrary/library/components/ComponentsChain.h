#pragma once

#include "INeurologicalComponent.h"
#include <vector>
#include <memory>

struct cuda_ComponentsChain : public cuda_INeurologicalComponent {
    std::vector<std::unique_ptr<cuda_INeurologicalComponent>> components;

    // Persistent buffers
    std::vector<cuda_Tensor> chainActivationBuffers;
    std::vector<cuda_Tensor> chainGradientBuffers;

    // Constructor
    cuda_ComponentsChain();

    // Static Operations
    static int getOutputRequiredSize(cuda_INeurologicalComponent* comp, const int totalSamples);
    static int getInputRequiredSize(cuda_INeurologicalComponent* comp, const int totalSamples);

    static void InitializeComponents(cuda_ComponentsChain& chain, unsigned long seed);

    static void ChainActivation(
        cuda_ComponentsChain& chain,
        cuda_ConstantTensorSpan input,
        cuda_TensorSpan output,
        const int& batchSize,
        lengths seriesLengths,
        const int& totalSamples
    );

    static void ChainInference(
        cuda_ComponentsChain& chain,
        cuda_ConstantTensorSpan input,
        cuda_TensorSpan output,
        const int& sequenceLength
    );
    static void ClearInference(
        cuda_ComponentsChain& layer
    );

    static void ChainAdjustment(
        cuda_ComponentsChain& chain,
        cuda_ConstantTensorSpan output_gradient,
        cuda_TensorSpan input_gradient,
        const int& batchSize,
        lengths seriesLengths,
        const int& totalSamples
    );
    
    static void ChainUpdate(
        cuda_ComponentsChain& chain,
        const cuda_neurologicalValue& updateMultiplier
    );
    
private:
    static void InitializeComponent(cuda_INeurologicalComponent* component, unsigned long seed);
    static unsigned long CountComponents(cuda_INeurologicalComponent* component);
};