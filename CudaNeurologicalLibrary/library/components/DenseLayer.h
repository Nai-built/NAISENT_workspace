#pragma once

#include "../activation_functions/IActivationFunction.h"
#include "../optimization/IOptimization.h"

#include <memory>
#include <random>

#include "INeurologicalComponent.h"

struct cuda_DenseLayer : public cuda_INeurologicalComponent {
private:
    int inputSize;
    int outputSize;
    int maxDropout;

public:
    
    // Activation Function
    std::unique_ptr<cuda_IActivationFunction> activation;
    
    // Parameters Optimization
    std::unique_ptr<cuda_IOptimization> weights_optimization;
    std::unique_ptr<cuda_IOptimization> biases_optimization;

    // Parameters
    cuda_Tensor weights;   // size = inputSize * outputSize  → layout: [in][out]
    cuda_Tensor biases;    // size = outputSize

    // Gradients
    cuda_Tensor weights_gradient; // size = inputSize * outputSize
    cuda_Tensor biases_gradient;  // size = outputSize

    // Cache
    cuda_ConstantTensorSpan cached_input;  // a span to the previously passed input

    // Extra
    cuda_Tensor dropout_gradient;
    cuda_SharedIntTensor dropout_mask;
    int droppedOut;
    std::mt19937 dropout_rand;
    std::uniform_int_distribution<int> dropout_rand_distribution;

    // Ctor / Dtor
    cuda_DenseLayer(int inSize, int outSize, int _maxDropout);

    // Size accessors
    int in_size() const;
    int out_size() const;

    // Logic
    // Static operations
    static void Initialize(cuda_DenseLayer& layer, unsigned long seed);

    static void ForwardPass(
        cuda_DenseLayer& layer,
        cuda_ConstantTensorSpan input,
        cuda_TensorSpan output,
        const int& samplesAmount
    );

    static void Infer(
        cuda_DenseLayer& layer,
        cuda_ConstantTensorSpan input,
        cuda_TensorSpan output,
        const int& sequenceLength
    );
    static void ClearInference(
        cuda_DenseLayer& layer
    );

    static void BackPropagation(
        cuda_DenseLayer& layer,
        cuda_ConstantTensorSpan output_gradient,
        cuda_TensorSpan input_gradient,
        const int& samplesAmount
    );
    
    static void Update(cuda_DenseLayer& layer, const cuda_neurologicalValue& updateMultiplier);
};