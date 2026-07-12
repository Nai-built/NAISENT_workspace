#include <iostream>

#include "ActivationFunctions/ReLU.h"

#include "DenseLayer.h"

#include "Extra.h"

DenseLayer::DenseLayer(int _inputSize, int _outputSize, ActivationFunction_UNIQUE _activationFunction) {
    this->componentType = NeurologicalComponentTypes::DENSE_LAYER;

    this->inputSize = _inputSize;
    this->outputSize = _outputSize;

    if (_activationFunction == nullptr) {
        this->activationFunction = std::make_unique<IActivationFunction>();
        this->activationFunction->activationType = ActivationFunctionTypes::LINEAR;
    } else {
        this->activationFunction = std::move(_activationFunction);
    }
}

void DenseLayer::Initialize(DenseLayer* denseLayer) {
    denseLayer->weights = neurologicalBuffer(denseLayer->outputSize*denseLayer->inputSize);
    denseLayer->biases = neurologicalBuffer(denseLayer->outputSize);

    denseLayer->weightsGradient = neurologicalBuffer(denseLayer->weights.size());
    denseLayer->biasesGradient = neurologicalBuffer(denseLayer->biases.size());

	neurologicalValue weights_initialization_boundary =
    (sqrt(2.0 / ((double)(denseLayer->inputSize)))*sqrt(3));
    // (neurologicalValue)sqrt(6 / ((double)(denseLayer->inputSize) + (double)(denseLayer->outputSize)));
	
    for (int i = 0; i < denseLayer->outputSize; i++) {
        denseLayer->biases[i] = 0.0f;
        for (int j = 0; j < denseLayer->inputSize; j++) {
            denseLayer->weights[j*(denseLayer->outputSize) + i] = getRandom(-weights_initialization_boundary, weights_initialization_boundary);
        }
    }
}

constexpr int batch_BLOCK = 64;
constexpr int output_BLOCK = 256;
constexpr int input_BLOCK = 256;

// forward pass
void dense_weights_kernel(const neurologicalValue* __restrict input, neurologicalValue* __restrict output, const neurologicalValue* __restrict weights
    , const int& batchSize, const int& inputSize, const int& outputSize)
{
    for (int batchStart = 0; batchStart < batchSize; batchStart+=batch_BLOCK) {
        int bmax = std::min(batchStart + batch_BLOCK, batchSize);
    for (int outputStart = 0; outputStart < outputSize; outputStart+=output_BLOCK) {
        int omax = std::min(outputStart + output_BLOCK, outputSize);
    for (int inputStart = 0; inputStart < inputSize; inputStart+=input_BLOCK) {
        int imax = std::min(inputStart + input_BLOCK, inputSize);

        for (int b = batchStart; b < bmax; ++b)
        {
            neurologicalValue* __restrict y = output + b * outputSize + outputStart;
            const neurologicalValue* __restrict x = input + b * inputSize;

            for (int i = inputStart; i < imax; ++i)
            {
                const neurologicalValue& xv = x[i];
                const neurologicalValue* __restrict w = weights + i * outputSize + outputStart;

                for (int o = 0; o < omax - outputStart; ++o)
                    y[o] += xv * w[o];
            }
        }
    }
    }
    }
}

// forward pass activations
void dense_activation_linear(neurologicalValue* __restrict output, const neurologicalValue* __restrict biases
    , const int& batchSize, const int& outputSize)
{
    for (int b = 0; b < batchSize; ++b)
    {
        neurologicalValue* __restrict y = output + b * outputSize;
        
        for (int o = 0; o < outputSize; ++o) {
            y[o] += biases[o];
        }
    }
}

inline neurologicalValue ReLU__result(const neurologicalValue& leakage, const neurologicalValue& v) {
    if (v < 0) {
        return v*leakage;
    }
    return v;
}
void dense_activation_relu(neurologicalValue* __restrict output, const neurologicalValue* __restrict biases
    , const int& batchSize, const int& outputSize
    , neurologicalValue* __restrict cached_output, const neurologicalValue& leakage)
{
    for (int b = 0; b < batchSize; ++b)
    {
        neurologicalValue* __restrict cached_y = cached_output + b * outputSize;
        neurologicalValue* __restrict y = output + b * outputSize;
        
        for (int o = 0; o < outputSize; ++o) {
            const neurologicalValue v = y[o] + biases[o];
            cached_y[o] = v;
            y[o] = ReLU__result(leakage, v);
            // printf("output: [%d,%d] = %f, also bias: %f\n", b, o, y[o], biases[outputStart + o]);
        }
    }
}


// back propagation activations
void dense_activation_gradient_linear(const neurologicalValue* __restrict propagation, neurologicalValue* __restrict biasesGradient
    , const int& batchSize, const int& outputSize)
{
    for (int b = 0; b < batchSize; ++b)
    {
        const neurologicalValue* __restrict p = propagation + b * outputSize;
        
        for (int o = 0; o < outputSize; ++o) {
            biasesGradient[o] += p[o];
        }
    }
}

inline neurologicalValue ReLU__gradient(const neurologicalValue& leakage, const neurologicalValue& v, const neurologicalValue& g) {
    if (v < 0) {
        return g*leakage;
    }
    return g;
}
void dense_activation_gradient_relu(neurologicalValue* __restrict propagation, neurologicalValue* __restrict biasesGradient
    , const int& batchSize, const int& outputSize
    , const neurologicalValue* __restrict cached_output, const neurologicalValue& leakage)
{
    for (int b = 0; b < batchSize; ++b)
    {
        const neurologicalValue* __restrict cached_y = cached_output + b * outputSize;

        neurologicalValue* __restrict p = propagation + b * outputSize;
        
        for (int o = 0; o < outputSize; ++o) {
            const neurologicalValue g = ReLU__gradient(leakage, cached_y[o], p[o]);
            p[o] = g;
            biasesGradient[o] += g;
        }
    }
}

// back propagation
void dense_propagation_input_kernel(const neurologicalValue* __restrict propagation, neurologicalValue* __restrict inputPropagation, const neurologicalValue* __restrict weights
    , const int& batchSize, const int& inputSize, const int& outputSize)
{
    for (int batchStart = 0; batchStart < batchSize; batchStart+=batch_BLOCK) {
        int bmax = std::min(batchStart + batch_BLOCK, batchSize);
    for (int outputStart = 0; outputStart < outputSize; outputStart+=output_BLOCK) {
        int omax = std::min(outputStart + output_BLOCK, outputSize);
    for (int inputStart = 0; inputStart < inputSize; inputStart+=input_BLOCK) {
        int imax = std::min(inputStart + input_BLOCK, inputSize);

        for (int b = batchStart; b < bmax; ++b)
        {
            neurologicalValue* inputP = inputPropagation + b * inputSize;

            const neurologicalValue* __restrict p = propagation + b * outputSize + outputStart;
            for (int i = inputStart; i < imax; ++i)
            {
                const neurologicalValue* __restrict w = weights + (i) * outputSize + outputStart;

                neurologicalValue g = 0.0f;

                for (int o = 0; o < omax - outputStart; ++o)
                    g += p[o] * w[o];

                inputP[i] += g;
            }
        }
    }
    }
    }
}

void dense_propagation_weights_kernel(const neurologicalValue* __restrict propagation, const neurologicalValue* __restrict cached_input, neurologicalValue* __restrict weightsGradient
    , const int& batchSize, const int& inputSize, const int& outputSize)
{
    for (int batchStart = 0; batchStart < batchSize; batchStart+=batch_BLOCK) {
        int bmax = std::min(batchStart + batch_BLOCK, batchSize);
    for (int outputStart = 0; outputStart < outputSize; outputStart+=output_BLOCK) {
        int omax = std::min(outputStart + output_BLOCK, outputSize);
    for (int inputStart = 0; inputStart < inputSize; inputStart+=input_BLOCK) {
        int imax = std::min(inputStart + input_BLOCK, inputSize);

        for (int b = batchStart; b < bmax; ++b)
        {
            const neurologicalValue* __restrict cachedX = cached_input + b * inputSize;

            const neurologicalValue* __restrict p = propagation + b * outputSize + outputStart;
            for (int i = inputStart; i < imax; ++i)
            {
                const neurologicalValue& _cachedX = cachedX[i];

                neurologicalValue* __restrict wG = weightsGradient + (i) * outputSize + outputStart;

                for (int o = 0; o < omax - outputStart; ++o) {
                    wG[o] += (p[o]*_cachedX);
                }
            }
        }
    }
    }
    }
}

// updating
void update_weights(neurologicalValue* __restrict weights, const neurologicalValue* __restrict weightsGradient
    , const int& inputSize, const int& outputSize
    , const neurologicalValue& rate)
{
    const neurologicalValue decayMultiplier = 1 - 0.001*rate;

    for (int i = 0; i < inputSize; ++i)
    {
        neurologicalValue* __restrict w  = weights + i * outputSize;
        const neurologicalValue* __restrict gw = weightsGradient + i * outputSize;

        for (int o = 0; o < outputSize; ++o)
            w[o] = w[o]*decayMultiplier - gw[o] * rate;
    }
}
void update_biases(neurologicalValue* __restrict biases, const neurologicalValue* __restrict biasesGradient
    , const int& outputSize
    , const neurologicalValue& rate)
{
    const neurologicalValue decayMultiplier = 1 - 0.001*rate;

    for (int o = 0; o < outputSize; ++o)
        biases[o] = biases[o]*decayMultiplier - biasesGradient[o] * rate;
}

void DenseLayer::ForwardPass(DenseLayer* denseLayer, const neurologicalConstantSpan input, const neurologicalSpan output, const int& batchSize) {
    // std::cout << input.size() << std::endl;

    // if (output.size() != batchSize*denseLayer.outputSize) {
    //     std::cout << "update output size properly" << std::endl;
    //     output = neurologicalBuffer(batchSize*denseLayer.outputSize);
    // }
    denseLayer->cached_input = input;
    // std::cout << "INPUT: " << denseLayer->cached_input[0] << std::endl;
    dense_weights_kernel(
        input.data(),
        output.data(),

        denseLayer->weights.data(),
        
        batchSize,
        denseLayer->inputSize,
        denseLayer->outputSize
    );
    switch(denseLayer->activationFunction->activationType) {
        case ActivationFunctionTypes::RELU: {
            ReLU* relu = static_cast<ReLU*>(denseLayer->activationFunction.get());
            if (relu->cached_output.size() != output.size()) {
                relu->cached_output.resize(output.size());
            }
            dense_activation_relu(
                output.data(),

                denseLayer->biases.data(),

                batchSize,
                denseLayer->outputSize,

                relu->cached_output.data(),
                relu->leakage
            );
            break;
        }
        default: {
            dense_activation_linear(
                output.data(),

                denseLayer->biases.data(),

                batchSize,
                denseLayer->outputSize
            );
            break;
        }
    }
}

void DenseLayer::BackPropagation(DenseLayer* denseLayer, const neurologicalSpan propagation, const neurologicalSpan inputPropagation, const neurologicalValue& learnRate, const int& batchSize) {
    neurologicalValue rate = learnRate/batchSize;

    switch(denseLayer->activationFunction->activationType) {
        case ActivationFunctionTypes::RELU: {
            ReLU* relu = static_cast<ReLU*>(denseLayer->activationFunction.get());
            dense_activation_gradient_relu(
                propagation.data(),
                denseLayer->biasesGradient.data(),

                batchSize,
                denseLayer->outputSize,

                relu->cached_output.data(),
                relu->leakage
            );
            break;
        }
        default: {
            dense_activation_gradient_linear(
                propagation.data(),
                denseLayer->biasesGradient.data(),

                batchSize,
                denseLayer->outputSize
            );
            break;
        }
    }
    dense_propagation_input_kernel(
        propagation.data(),
        inputPropagation.data(),

        // unmodified weights
        denseLayer->weights.data(),

        batchSize,
        denseLayer->inputSize,
        denseLayer->outputSize
    );
    dense_propagation_weights_kernel(
        propagation.data(),
        denseLayer->cached_input.data(),

        denseLayer->weightsGradient.data(),

        batchSize,
        denseLayer->inputSize,
        denseLayer->outputSize
    );

    update_biases(
        denseLayer->biases.data(),
        denseLayer->biasesGradient.data(),

        denseLayer->outputSize,

        rate
    );
    update_weights(
        denseLayer->weights.data(),
        denseLayer->weightsGradient.data(),

        denseLayer->inputSize,
        denseLayer->outputSize,

        rate
    );

    zeroOutList(denseLayer->weightsGradient);
    zeroOutList(denseLayer->biasesGradient);
}