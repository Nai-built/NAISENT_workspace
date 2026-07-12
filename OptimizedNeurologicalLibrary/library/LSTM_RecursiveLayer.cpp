#include <iostream>

#include "LSTM_RecursiveLayer.h"

constexpr int LSTM_UNIT_GATES = 4;

LSTM_RecursiveLayer::LSTM_RecursiveLayer(int _inputSize, int _outputSize, bool _castBeyond) {
    this->componentType = NeurologicalComponentTypes::RECURSIVE_LSTM;

    this->inputSize = _inputSize;
    this->outputSize = _outputSize;
    
    this->castBeyond = _castBeyond;
}

void LSTM_RecursiveLayer::Initialize(LSTM_RecursiveLayer* lstm) {
    lstm->maxSamplesOutputSize = 0;

    lstm->hiddenStatePropagationBuffer = neurologicalBuffer(lstm->outputSize);
    lstm->memoryCellPropagationBuffer = neurologicalBuffer(lstm->outputSize);

    const int gatesAmount = lstm->outputSize*LSTM_UNIT_GATES;
    const int gatesInput = lstm->outputSize+lstm->inputSize;

    lstm->weights = neurologicalBuffer(gatesAmount*gatesInput);
    lstm->biases = neurologicalBuffer(gatesAmount);

    lstm->weightsGradient = neurologicalBuffer(lstm->weights.size());
    lstm->biasesGradient = neurologicalBuffer(lstm->biases.size());

	neurologicalValue weights_initialization_boundary =
    (sqrt(2.0 / ((double)(gatesInput)))*sqrt(3));
    // (neurologicalValue)sqrt(6 / ((double)(gatesInput) + (double)(gatesAmount)));
	
    for (int i = 0; i < gatesAmount; i++) {
        lstm->biases[i] = 0.0f;
    }
    for (int o = 0; o < lstm->outputSize; o++) {
        for (int i = 0; i < gatesInput; i++) {
            for (int g = 0; g < LSTM_UNIT_GATES; g++) {
                lstm->weights[o*gatesInput*LSTM_UNIT_GATES + i*LSTM_UNIT_GATES + g]
                    = getRandom(-weights_initialization_boundary, weights_initialization_boundary);
            }
        }
    }
}

// recursive system for lstm:
// each batch sample would be actually a "series" of samples
// for example: a batch of 4 lstm inputs would be [[3, 2, 1], [4, 1], [6], [5, 7, 9, 0]] but obviously vectorized (in this case the input size is just 1 value)
// the lstm would process every sample in this series properly then output a single vector output for each sample series in the batch (like normal dense layers)
// what if that thing called "castBeyond?": we would make it the lstm can take actions while still recording its memeory to use for later steps, however, thats just for later.

// TO DO: make batch size assignable from the controller and not calculated by our layers!

inline neurologicalValue tanh_derivative(const neurologicalValue& activation)
{
	return ((double)1) - pow(activation, 2);
}
inline neurologicalValue sigmoid_derivative(const neurologicalValue& activation) {
    return (activation*(((double)1)-activation));
}
inline void sigmoid(neurologicalValue& v) {
    v = ((double)1.0) / (((double)1.0) + std::exp(-v));
}
void lstm_inner(const neurologicalValue* __restrict input, neurologicalValue* __restrict output
    , const neurologicalValue* __restrict weights, const neurologicalValue* __restrict biases
    , const int& batchSize, const int* __restrict seriesLengths
    , const int& inputSize, const int& outputSize
    
    , neurologicalValue* __restrict hidden_state_buffer
    , neurologicalValue* __restrict memory_cell_buffer

    , neurologicalValue* __restrict forgetGate_buffer
    , neurologicalValue* __restrict inputGate_buffer
    , neurologicalValue* __restrict memoryGate_buffer
    , neurologicalValue* __restrict outputGate_buffer

    , neurologicalValue* __restrict memoryCell_activation)
{
    for (int b = 0; b < batchSize; ++b)
    {
        const int& seriesLength = seriesLengths[b];

        for (int s = 0; s < seriesLength; ++s) {
            for (int o = 0; o < outputSize; ++o) {
                // compute lstm unit:

                const neurologicalValue* __restrict w = weights + o*(outputSize+inputSize)*LSTM_UNIT_GATES;

                const neurologicalValue* __restrict unitBiases = biases + o*LSTM_UNIT_GATES;

                neurologicalValue forgetGate = unitBiases[0];
                neurologicalValue inputGate = unitBiases[1];
                neurologicalValue memoryGate = unitBiases[2];
                neurologicalValue outputGate = unitBiases[3];

                for (int h = 0; h < outputSize; ++h) {
                    const neurologicalValue& hiddenValue = hidden_state_buffer[h];

                    forgetGate += hiddenValue*w[0];
                    inputGate += hiddenValue*w[1];
                    memoryGate += hiddenValue*w[2];
                    outputGate += hiddenValue*w[3];

                    w+=LSTM_UNIT_GATES;
                }
                for (int i = 0; i < inputSize; ++i) {
                    const neurologicalValue& inputValue = input[i];

                    forgetGate += inputValue*w[0];
                    inputGate += inputValue*w[1];
                    memoryGate += inputValue*w[2];
                    outputGate += inputValue*w[3];
                    
                    w+=LSTM_UNIT_GATES;
                }

                sigmoid(forgetGate);
                sigmoid(inputGate);
                sigmoid(outputGate);
                
                forgetGate_buffer[o] = forgetGate;
                inputGate_buffer[o] = inputGate;
                memoryGate_buffer[o] = tanh(memoryGate);
                outputGate_buffer[o] = outputGate;
            }

            // take last produced hidden state as output
            if (s >= seriesLength-1) {
                for (int o = 0; o < outputSize; ++o) {
                    const neurologicalValue memory_cell_value = (memory_cell_buffer[o]*forgetGate_buffer[o])
                        + (inputGate_buffer[o]*memoryGate_buffer[o]);
                    const neurologicalValue memory_cell_value_tanh = tanh(memory_cell_value);
                    memoryCell_activation[o] = memory_cell_value_tanh;

                    output[o] = outputGate_buffer[o]*memory_cell_value_tanh;
                }
            }
            // update memory after series step
            else {
                for (int o = 0; o < outputSize; ++o) {
                    const neurologicalValue memory_cell_value = (memory_cell_buffer[o]*forgetGate_buffer[o])
                        + (inputGate_buffer[o]*memoryGate_buffer[o]);
                    const neurologicalValue memory_cell_value_tanh = tanh(memory_cell_value);
                    memoryCell_activation[o] = memory_cell_value_tanh;

                    hidden_state_buffer[o+outputSize] = outputGate_buffer[o]*memory_cell_value_tanh;
                    memory_cell_buffer[o+outputSize] = memory_cell_value;
                }
            }

            hidden_state_buffer += outputSize;
            memory_cell_buffer += outputSize;
            
            forgetGate_buffer += outputSize;
            inputGate_buffer += outputSize;
            memoryGate_buffer += outputSize;
            outputGate_buffer += outputSize;

            memoryCell_activation += outputSize;

            input += inputSize;
        }
        output += outputSize;
    }
}
void lstm_beyond(const neurologicalValue* __restrict input, neurologicalValue* __restrict output
    , const neurologicalValue* __restrict weights, const neurologicalValue* __restrict biases
    , const int& batchSize, const int* __restrict seriesLengths
    , const int& inputSize, const int& outputSize
    
    , neurologicalValue* __restrict hidden_state_buffer
    , neurologicalValue* __restrict memory_cell_buffer

    , neurologicalValue* __restrict forgetGate_buffer
    , neurologicalValue* __restrict inputGate_buffer
    , neurologicalValue* __restrict memoryGate_buffer
    , neurologicalValue* __restrict outputGate_buffer

    , neurologicalValue* __restrict memoryCell_activation)
{
    for (int b = 0; b < batchSize; ++b)
    {
        const int& seriesLength = seriesLengths[b];

        for (int s = 0; s < seriesLength; ++s) {
            for (int o = 0; o < outputSize; ++o) {
                // compute lstm unit:

                const neurologicalValue* __restrict w = weights + o*(outputSize+inputSize)*LSTM_UNIT_GATES;

                const neurologicalValue* __restrict unitBiases = biases + o*LSTM_UNIT_GATES;

                neurologicalValue forgetGate = unitBiases[0];
                neurologicalValue inputGate = unitBiases[1];
                neurologicalValue memoryGate = unitBiases[2];
                neurologicalValue outputGate = unitBiases[3];

                for (int h = 0; h < outputSize; ++h) {
                    const neurologicalValue& hiddenValue = hidden_state_buffer[h];

                    forgetGate += hiddenValue*w[0];
                    inputGate += hiddenValue*w[1];
                    memoryGate += hiddenValue*w[2];
                    outputGate += hiddenValue*w[3];

                    w+=LSTM_UNIT_GATES;
                }
                for (int i = 0; i < inputSize; ++i) {
                    const neurologicalValue& inputValue = input[i];

                    forgetGate += inputValue*w[0];
                    inputGate += inputValue*w[1];
                    memoryGate += inputValue*w[2];
                    outputGate += inputValue*w[3];
                    
                    w+=LSTM_UNIT_GATES;
                }

                sigmoid(forgetGate);
                sigmoid(inputGate);
                sigmoid(outputGate);
                
                forgetGate_buffer[o] = forgetGate;
                inputGate_buffer[o] = inputGate;
                memoryGate_buffer[o] = tanh(memoryGate);
                outputGate_buffer[o] = outputGate;
            }

            // take last produced hidden state as output
            if (s >= seriesLength-1) {
                for (int o = 0; o < outputSize; ++o) {
                    const neurologicalValue memory_cell_value = (memory_cell_buffer[o]*forgetGate_buffer[o])
                        + (inputGate_buffer[o]*memoryGate_buffer[o]);
                    const neurologicalValue memory_cell_value_tanh = tanh(memory_cell_value);
                    memoryCell_activation[o] = memory_cell_value_tanh;

                    output[o] = outputGate_buffer[o]*memory_cell_value_tanh;
                }
            }
            // update memory after series step
            else {
                for (int o = 0; o < outputSize; ++o) {
                    const neurologicalValue memory_cell_value = (memory_cell_buffer[o]*forgetGate_buffer[o])
                        + (inputGate_buffer[o]*memoryGate_buffer[o]);
                    const neurologicalValue memory_cell_value_tanh = tanh(memory_cell_value);
                    memoryCell_activation[o] = memory_cell_value_tanh;

                    const neurologicalValue outputValue = outputGate_buffer[o]*memory_cell_value_tanh;

                    hidden_state_buffer[o+outputSize] = outputValue;
                    memory_cell_buffer[o+outputSize] = memory_cell_value;

                    output[o] = outputValue;
                }
            }

            hidden_state_buffer += outputSize;
            memory_cell_buffer += outputSize;
            
            forgetGate_buffer += outputSize;
            inputGate_buffer += outputSize;
            memoryGate_buffer += outputSize;
            outputGate_buffer += outputSize;

            memoryCell_activation += outputSize;

            input += inputSize;

            output += outputSize;
        }
    }
}
void lstm_inner_propagation(neurologicalValue* __restrict propagation, neurologicalValue* __restrict inputPropagation
    , neurologicalValue* __restrict hiddenStatePropagationBuffer, neurologicalValue* __restrict memoryCellPropagationBuffer
    , const neurologicalValue* __restrict cached_input
    , const neurologicalValue* __restrict weights
    , neurologicalValue* __restrict weightsGradient, neurologicalValue* __restrict biasesGradient
    , const int& batchSize, const int* __restrict seriesLengths
    , const int& inputSize, const int& outputSize
    
    , const neurologicalValue* __restrict cached_hidden_state_buffer
    , const neurologicalValue* __restrict cached_memory_cell_buffer

    , const neurologicalValue* __restrict cached_forgetGate_buffer
    , const neurologicalValue* __restrict cached_inputGate_buffer
    , const neurologicalValue* __restrict cached_memoryGate_buffer
    , const neurologicalValue* __restrict cached_outputGate_buffer

    , const neurologicalValue* __restrict cached_memoryCell_activation)
{
    for (int b = 0; b < batchSize; ++b)
    {
        neurologicalValue* __restrict p = propagation + b * outputSize;

        const int& seriesLength = seriesLengths[b];

        for (int s = seriesLength-1; s >= 0; --s) {
            // prepare cached information
            const neurologicalValue* __restrict x = cached_input + s * inputSize;
            neurologicalValue* __restrict xP = inputPropagation + s * inputSize;

            const int unitOutputPoint = s * outputSize;

            const neurologicalValue* __restrict hidden_state = cached_hidden_state_buffer + unitOutputPoint;
            const neurologicalValue* __restrict memory_cell = cached_memory_cell_buffer + unitOutputPoint;

            const neurologicalValue* __restrict forgetGate_buffer = cached_forgetGate_buffer + unitOutputPoint;
            const neurologicalValue* __restrict inputGate_buffer = cached_inputGate_buffer + unitOutputPoint;
            const neurologicalValue* __restrict memoryGate_buffer = cached_memoryGate_buffer + unitOutputPoint;
            const neurologicalValue* __restrict outputGate_buffer = cached_outputGate_buffer + unitOutputPoint;

            const neurologicalValue* __restrict memoryCell_activation = cached_memoryCell_activation + unitOutputPoint;
            
            for (int o = 0; o < outputSize; ++o) {
                const int wIndex = o*(outputSize+inputSize)*LSTM_UNIT_GATES;

                const neurologicalValue* __restrict w = weights + wIndex;
                neurologicalValue* __restrict wG = weightsGradient + wIndex;

                neurologicalValue* __restrict unitBiasesG = biasesGradient + o*LSTM_UNIT_GATES;

                // gather all propagation values properly
                const neurologicalValue& propagationValue = p[o];

                neurologicalValue outputGatePropagationValue = propagationValue*memoryCell_activation[o]
                    * sigmoid_derivative(outputGate_buffer[o]);

                neurologicalValue memoryCellPropagationValue = (propagationValue*outputGate_buffer[o]
                    * tanh_derivative(memoryCell_activation[o]))
                    + memoryCellPropagationBuffer[o];

                neurologicalValue forgetGatePropagationValue = memoryCellPropagationValue*memory_cell[o]
                    * sigmoid_derivative(forgetGate_buffer[o]);
                
                neurologicalValue inputGatePropagationValue = memoryCellPropagationValue*memoryGate_buffer[o]
                    * sigmoid_derivative(inputGate_buffer[o]);

                neurologicalValue memoryGatePropagationValue = memoryCellPropagationValue*inputGate_buffer[o]
                    * tanh_derivative(memoryGate_buffer[o]);

                // process propagation
                memoryCellPropagationBuffer[o] = memoryCellPropagationValue*forgetGate_buffer[o];
                
                unitBiasesG[0] += forgetGatePropagationValue;
                unitBiasesG[1] += inputGatePropagationValue;
                unitBiasesG[2] += memoryGatePropagationValue;
                unitBiasesG[3] += outputGatePropagationValue;

                for (int h = 0; h < outputSize; ++h) {
                    const neurologicalValue& hiddenValue = hidden_state[h];

                    neurologicalValue hiddenPropagation = 0.0;

                    hiddenPropagation += forgetGatePropagationValue*w[0];
                    hiddenPropagation += inputGatePropagationValue*w[1];
                    hiddenPropagation += memoryGatePropagationValue*w[2];
                    hiddenPropagation += outputGatePropagationValue*w[3];

                    hiddenStatePropagationBuffer[h] += hiddenPropagation;

                    wG[0] += hiddenValue*forgetGatePropagationValue;
                    wG[1] += hiddenValue*inputGatePropagationValue;
                    wG[2] += hiddenValue*memoryGatePropagationValue;
                    wG[3] += hiddenValue*outputGatePropagationValue;

                    w+=LSTM_UNIT_GATES;
                    wG+=LSTM_UNIT_GATES;
                }
                for (int i = 0; i < inputSize; ++i) {
                    const neurologicalValue& inputValue = x[i];

                    neurologicalValue inputPropagationValue = 0.0;

                    inputPropagationValue += forgetGatePropagationValue*w[0];
                    inputPropagationValue += inputGatePropagationValue*w[1];
                    inputPropagationValue += memoryGatePropagationValue*w[2];
                    inputPropagationValue += outputGatePropagationValue*w[3];

                    xP[i] += inputPropagationValue;

                    wG[0] += inputValue*forgetGatePropagationValue;
                    wG[1] += inputValue*inputGatePropagationValue;
                    wG[2] += inputValue*memoryGatePropagationValue;
                    wG[3] += inputValue*outputGatePropagationValue;

                    w+=LSTM_UNIT_GATES;
                    wG+=LSTM_UNIT_GATES;
                }
            }

            for (int o = 0; o < outputSize; ++o) {
                p[o] = hiddenStatePropagationBuffer[o];
                hiddenStatePropagationBuffer[o] = 0;
            }
            
            // std::memset(hiddenStatePropagationBuffer, 0, outputSize*sizeof(neurologicalValue));
        }

        std::memset(memoryCellPropagationBuffer, 0, outputSize*sizeof(neurologicalValue));

        const int outputSpace = seriesLength*outputSize;

        cached_hidden_state_buffer += outputSpace;
        cached_memory_cell_buffer += outputSpace;
            
        cached_forgetGate_buffer += outputSpace;
        cached_inputGate_buffer += outputSpace;
        cached_memoryGate_buffer += outputSpace;
        cached_outputGate_buffer += outputSpace;

        cached_memoryCell_activation += outputSpace;

        cached_input += seriesLength*inputSize;
        inputPropagation += seriesLength*inputSize;
    }
}
void lstm_beyond_propagation(neurologicalValue* __restrict propagation, neurologicalValue* __restrict inputPropagation
    , neurologicalValue* __restrict hiddenStatePropagationBuffer, neurologicalValue* __restrict memoryCellPropagationBuffer
    , const neurologicalValue* __restrict cached_input
    , const neurologicalValue* __restrict weights
    , neurologicalValue* __restrict weightsGradient, neurologicalValue* __restrict biasesGradient
    , const int& batchSize, const int* __restrict seriesLengths
    , const int& inputSize, const int& outputSize
    
    , const neurologicalValue* __restrict cached_hidden_state_buffer
    , const neurologicalValue* __restrict cached_memory_cell_buffer

    , const neurologicalValue* __restrict cached_forgetGate_buffer
    , const neurologicalValue* __restrict cached_inputGate_buffer
    , const neurologicalValue* __restrict cached_memoryGate_buffer
    , const neurologicalValue* __restrict cached_outputGate_buffer

    , const neurologicalValue* __restrict cached_memoryCell_activation)
{
    for (int b = 0; b < batchSize; ++b)
    {
        const int& seriesLength = seriesLengths[b];

        for (int s = seriesLength-1; s >= 0; --s) {
            // prepare cached information
            const neurologicalValue* __restrict x = cached_input + s * inputSize;
            neurologicalValue* __restrict xP = inputPropagation + s * inputSize;

            const int unitOutputPoint = s * outputSize;

            neurologicalValue* __restrict p = propagation + unitOutputPoint;

            const neurologicalValue* __restrict hidden_state = cached_hidden_state_buffer + unitOutputPoint;
            const neurologicalValue* __restrict memory_cell = cached_memory_cell_buffer + unitOutputPoint;

            const neurologicalValue* __restrict forgetGate_buffer = cached_forgetGate_buffer + unitOutputPoint;
            const neurologicalValue* __restrict inputGate_buffer = cached_inputGate_buffer + unitOutputPoint;
            const neurologicalValue* __restrict memoryGate_buffer = cached_memoryGate_buffer + unitOutputPoint;
            const neurologicalValue* __restrict outputGate_buffer = cached_outputGate_buffer + unitOutputPoint;

            const neurologicalValue* __restrict memoryCell_activation = cached_memoryCell_activation + unitOutputPoint;
            
            for (int o = 0; o < outputSize; ++o) {
                const int wIndex = o*(outputSize+inputSize)*LSTM_UNIT_GATES;

                const neurologicalValue* __restrict w = weights + wIndex;
                neurologicalValue* __restrict wG = weightsGradient + wIndex;

                neurologicalValue* __restrict unitBiasesG = biasesGradient + o*LSTM_UNIT_GATES;

                // gather all propagation values properly
                const neurologicalValue& propagationValue = p[o];

                neurologicalValue outputGatePropagationValue = propagationValue*memoryCell_activation[o]
                    * sigmoid_derivative(outputGate_buffer[o]);

                neurologicalValue memoryCellPropagationValue = (propagationValue*outputGate_buffer[o]
                    * tanh_derivative(memoryCell_activation[o]))
                    + memoryCellPropagationBuffer[o];

                neurologicalValue forgetGatePropagationValue = memoryCellPropagationValue*memory_cell[o]
                    * sigmoid_derivative(forgetGate_buffer[o]);
                
                neurologicalValue inputGatePropagationValue = memoryCellPropagationValue*memoryGate_buffer[o]
                    * sigmoid_derivative(inputGate_buffer[o]);

                neurologicalValue memoryGatePropagationValue = memoryCellPropagationValue*inputGate_buffer[o]
                    * tanh_derivative(memoryGate_buffer[o]);

                // process propagation
                memoryCellPropagationBuffer[o] = memoryCellPropagationValue*forgetGate_buffer[o];
                
                unitBiasesG[0] += forgetGatePropagationValue;
                unitBiasesG[1] += inputGatePropagationValue;
                unitBiasesG[2] += memoryGatePropagationValue;
                unitBiasesG[3] += outputGatePropagationValue;

                for (int h = 0; h < outputSize; ++h) {
                    const neurologicalValue& hiddenValue = hidden_state[h];

                    neurologicalValue hiddenPropagation = 0.0;

                    hiddenPropagation += forgetGatePropagationValue*w[0];
                    hiddenPropagation += inputGatePropagationValue*w[1];
                    hiddenPropagation += memoryGatePropagationValue*w[2];
                    hiddenPropagation += outputGatePropagationValue*w[3];

                    hiddenStatePropagationBuffer[h] += hiddenPropagation;

                    wG[0] += hiddenValue*forgetGatePropagationValue;
                    wG[1] += hiddenValue*inputGatePropagationValue;
                    wG[2] += hiddenValue*memoryGatePropagationValue;
                    wG[3] += hiddenValue*outputGatePropagationValue;

                    w+=LSTM_UNIT_GATES;
                    wG+=LSTM_UNIT_GATES;
                }
                for (int i = 0; i < inputSize; ++i) {
                    const neurologicalValue& inputValue = x[i];

                    neurologicalValue inputPropagationValue = 0.0;

                    inputPropagationValue += forgetGatePropagationValue*w[0];
                    inputPropagationValue += inputGatePropagationValue*w[1];
                    inputPropagationValue += memoryGatePropagationValue*w[2];
                    inputPropagationValue += outputGatePropagationValue*w[3];

                    xP[i] += inputPropagationValue;

                    wG[0] += inputValue*forgetGatePropagationValue;
                    wG[1] += inputValue*inputGatePropagationValue;
                    wG[2] += inputValue*memoryGatePropagationValue;
                    wG[3] += inputValue*outputGatePropagationValue;

                    w+=LSTM_UNIT_GATES;
                    wG+=LSTM_UNIT_GATES;
                }
            }

            if (s > 0) {
                for (int o = 0; o < outputSize; ++o) {
                    p[o-outputSize] += hiddenStatePropagationBuffer[o];
                }
            }
            
            std::memset(hiddenStatePropagationBuffer, 0, outputSize*sizeof(neurologicalValue));
        }

        std::memset(memoryCellPropagationBuffer, 0, outputSize*sizeof(neurologicalValue));

        const int outputSpace = seriesLength*outputSize;

        cached_hidden_state_buffer += outputSpace;
        cached_memory_cell_buffer += outputSpace;
            
        cached_forgetGate_buffer += outputSpace;
        cached_inputGate_buffer += outputSpace;
        cached_memoryGate_buffer += outputSpace;
        cached_outputGate_buffer += outputSpace;

        cached_memoryCell_activation += outputSpace;

        propagation += outputSpace;

        cached_input += seriesLength*inputSize;
        inputPropagation += seriesLength*inputSize;
    }
}

inline void ensureBuffers(size_t needed, size_t& size,
    neurologicalBuffer& hidden_state_buffer,
    neurologicalBuffer& memory_cell_buffer,

    neurologicalBuffer& forgetGate_buffer,
    neurologicalBuffer& inputGate_buffer,
    neurologicalBuffer& memoryGate_buffer,
    neurologicalBuffer& outputGate_buffer,

    neurologicalBuffer& memoryCell_activation)
{
    if (size >= needed) return;

    hidden_state_buffer.resize(needed);
    memory_cell_buffer.resize(needed);
    
    forgetGate_buffer.resize(needed);
    inputGate_buffer.resize(needed);
    memoryGate_buffer.resize(needed);
    outputGate_buffer.resize(needed);
    
    memoryCell_activation.resize(needed);

    size = needed;
}

// updating
void update_weight_LSTM(neurologicalValue* __restrict weights, const neurologicalValue* __restrict weightsGradient
    , const int& gradientSize
    , const neurologicalValue& rate)
{
    for (int i = 0; i < gradientSize; ++i)
    {
        // std::cout << "weight gradient value: " << weightsGradient[i] << std::endl;
        // if (weightsGradient[i] > 25) {
        //     std::cout << "weight gradient value: " << weightsGradient[i] << std::endl;

        //     // throw std::runtime_error("LARGE WEIGHTS GRADIENT FOR LSTM!");
        // }
        weights[i] -= weightsGradient[i] * rate;
    }
}
void update_biases_LSTM(neurologicalValue* __restrict biases, const neurologicalValue* __restrict biasesGradient
    , const int& gradientSize
    , const neurologicalValue& rate)
{
    for (int i = 0; i < gradientSize; ++i)
        biases[i] -= biasesGradient[i] * rate;
}

void LSTM_RecursiveLayer::ForwardPass(LSTM_RecursiveLayer* lstm, const neurologicalConstantSpan input, const neurologicalSpan output, const int& batchSize
    , const lengths seriesLengths, const int& totalSamples)
{
    lstm->cached_input = input;

    ensureBuffers(totalSamples*lstm->outputSize, lstm->maxSamplesOutputSize,
        lstm->hidden_state_buffer,
        lstm->memory_cell_buffer,

        lstm->forgetGate_buffer,
        lstm->inputGate_buffer,
        lstm->memoryGate_buffer,
        lstm->outputGate_buffer,
        
        lstm->memoryCell_activation);

    zeroOutList(lstm->hidden_state_buffer);
    zeroOutList(lstm->memory_cell_buffer);

    zeroOutList(lstm->forgetGate_buffer);
    zeroOutList(lstm->inputGate_buffer);
    zeroOutList(lstm->memoryGate_buffer);
    zeroOutList(lstm->outputGate_buffer);
    
    zeroOutList(lstm->memoryCell_activation);
    
    if (lstm->castBeyond) {
        lstm_beyond(
            input.data(),
            output.data(),
            lstm->weights.data(),
            lstm->biases.data(),
            batchSize,
            seriesLengths.data(),
            lstm->inputSize,
            lstm->outputSize,

            lstm->hidden_state_buffer.data(),
            lstm->memory_cell_buffer.data(),

            lstm->forgetGate_buffer.data(),
            lstm->inputGate_buffer.data(),
            lstm->memoryGate_buffer.data(),
            lstm->outputGate_buffer.data(),
            
            lstm->memoryCell_activation.data()
        );
    } else {
        lstm_inner(
            input.data(),
            output.data(),
            lstm->weights.data(),
            lstm->biases.data(),
            batchSize,
            seriesLengths.data(),
            lstm->inputSize,
            lstm->outputSize,

            lstm->hidden_state_buffer.data(),
            lstm->memory_cell_buffer.data(),

            lstm->forgetGate_buffer.data(),
            lstm->inputGate_buffer.data(),
            lstm->memoryGate_buffer.data(),
            lstm->outputGate_buffer.data(),
            
            lstm->memoryCell_activation.data()
        );
    }
}
void LSTM_RecursiveLayer::BackPropagation(LSTM_RecursiveLayer* lstm, const neurologicalSpan propagation, const neurologicalSpan inputPropagation, const neurologicalValue& learnRate, const int& batchSize
    , const lengths seriesLengths, const int& totalSamples)
{
    neurologicalValue rate = learnRate/batchSize;

    ensureBuffers(totalSamples*lstm->outputSize, lstm->maxSamplesOutputSize,
        lstm->hidden_state_buffer,
        lstm->memory_cell_buffer,

        lstm->forgetGate_buffer,
        lstm->inputGate_buffer,
        lstm->memoryGate_buffer,
        lstm->outputGate_buffer,
        
        lstm->memoryCell_activation);

    if (lstm->castBeyond) {
        lstm_beyond_propagation(
            propagation.data(),
            inputPropagation.data(),
            lstm->hiddenStatePropagationBuffer.data(),
            lstm->memoryCellPropagationBuffer.data(),
            lstm->cached_input.data(),
            lstm->weights.data(),
            lstm->weightsGradient.data(),
            lstm->biasesGradient.data(),
            batchSize,
            seriesLengths.data(),
            lstm->inputSize,
            lstm->outputSize,

            lstm->hidden_state_buffer.data(),
            lstm->memory_cell_buffer.data(),

            lstm->forgetGate_buffer.data(),
            lstm->inputGate_buffer.data(),
            lstm->memoryGate_buffer.data(),
            lstm->outputGate_buffer.data(),
            
            lstm->memoryCell_activation.data()
        );
    } else {
        lstm_inner_propagation(
            propagation.data(),
            inputPropagation.data(),
            lstm->hiddenStatePropagationBuffer.data(),
            lstm->memoryCellPropagationBuffer.data(),
            lstm->cached_input.data(),
            lstm->weights.data(),
            lstm->weightsGradient.data(),
            lstm->biasesGradient.data(),
            batchSize,
            seriesLengths.data(),
            lstm->inputSize,
            lstm->outputSize,

            lstm->hidden_state_buffer.data(),
            lstm->memory_cell_buffer.data(),

            lstm->forgetGate_buffer.data(),
            lstm->inputGate_buffer.data(),
            lstm->memoryGate_buffer.data(),
            lstm->outputGate_buffer.data(),
            
            lstm->memoryCell_activation.data()
        );
    }

    update_biases_LSTM(
        lstm->biases.data(),
        lstm->biasesGradient.data(),

        lstm->biases.size(),

        rate
    );
    update_weight_LSTM(
        lstm->weights.data(),
        lstm->weightsGradient.data(),

        lstm->weights.size(),

        rate
    );

    zeroOutList(lstm->weightsGradient);
    zeroOutList(lstm->biasesGradient);
}