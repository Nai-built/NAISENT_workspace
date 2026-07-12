#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cuda_Tensor.h"

#include "configuration/BuildConfigurations.h"
#include "configuration/SaveConfigurations.h"

// Build entry
std::string BuildModel(const ComponentsChain__BuildInfo& buildInfo, const BaseOptimizer__BuildInfo& optInfo);

// Extract model parameters to CPU
std::shared_ptr<BaseNeurologicalComponent__SaveInfo> ExtractModel(const std::string& modelId);

// Insert model parameters from CPU
bool InsertModel(const std::string& modelId,
                 const std::shared_ptr<BaseNeurologicalComponent__SaveInfo>& saveInfo);

// Initialize
void InitializeModel(const std::string& modelId, unsigned long seed);

// Infer
void InferModel(
    const std::string& modelId,
    pybind11::buffer inputBuffer,
    pybind11::buffer outputBuffer,
    const int& sequenceLength
);
void ClearModelInference(
    const std::string& modelId
);

// Activate
void ActivateModel(
    const std::string& modelId,
    pybind11::buffer inputBuffer,
    pybind11::buffer outputBuffer,
    const int& batchSize,
    pybind11::buffer seriesLengths,
    const int& totalSamples
);

// Backprop
void AdjustModel(
    const std::string& modelId,
    pybind11::buffer outputGradBuffer,
    pybind11::buffer inputGradBuffer,
    const int& batchSize,
    pybind11::buffer seriesLengths,
    const int& totalSamples
);

// Update
void UpdateModel(
    const std::string& modelId,
    const cuda_neurologicalValue& learningAlpha
);

void MSE(pybind11::buffer prediction, pybind11::buffer correction, pybind11::buffer propagation);

void softmax(pybind11::buffer values, const int& eachSize);
// void testActivation(const cuda_neurologicalConstantSpan input, const cuda_neurologicalSpan output, const int& batchSize);