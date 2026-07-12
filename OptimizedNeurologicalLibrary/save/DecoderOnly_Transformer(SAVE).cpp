#include <iostream>

#include "DecoderOnly_Transformer(SAVE).h"

DecoderOnly_Transformer::DecoderOnly_Transformer(int _inputSize, int _outputSize, int _encodingSize) {
    this->componentType = NeurologicalComponentTypes::DECODER_ONLY_TRANSFORMER;

    this->inputSize = _inputSize;
    this->outputSize = _outputSize;
    this->encodingSize = _encodingSize;
}

void DecoderOnly_Transformer::Initialize(DecoderOnly_Transformer* transformer) {
    int weightsAmount = 0;

    // token encoding weights amount
    weightsAmount += transformer->inputSize * transformer->encodingSize;

    // key weights amount
    weightsAmount += transformer->encodingSize * transformer->encodingSize;
    // value weights amount
    weightsAmount += transformer->encodingSize * transformer->encodingSize;
    // query weights amount
    weightsAmount += transformer->encodingSize * transformer->encodingSize;
    
    // decoding weights amount
    weightsAmount += transformer->outputSize * transformer->encodingSize;

    const int decodingBiasesAmount = transformer->outputSize;

    transformer->weights = neurologicalBuffer(weightsAmount);
    transformer->biases = neurologicalBuffer(decodingBiasesAmount);

    transformer->weightsGradient = neurologicalBuffer(transformer->weights.size());
    transformer->biasesGradient = neurologicalBuffer(transformer->biases.size());
    
	neurologicalValue weights_initialization_boundary =
    (sqrt(2.0 / ((double)(transformer->inputSize)))*sqrt(3));

    for (int i = 0; i < transformer->weights.size(); i++) {
        transformer->weights[i]
        = getRandom(-weights_initialization_boundary, weights_initialization_boundary);
    }
}

inline void softmax(const neurologicalValue* __restrict input, neurologicalValue* __restrict output, const int& size) {
    neurologicalValue expSum = 0.0;

    for (int i = 0; i < size; ++i) {
        expSum += exp(input[i]);
    }

    for (int i = 0; i < size; ++i) {
        output[i] = exp(input[i])/expSum;
    }
}

// help functions
inline void encode_input(const neurologicalValue* __restrict input, neurologicalValue* __restrict encodedInput
    , const neurologicalValue* __restrict weights
    , const int& inputSize, const int& encodingSize
    , const int& pos)
{
    for (int i = 0; i < inputSize; ++i) {
        const neurologicalValue& inputValue = input[i];

        const neurologicalValue* __restrict w = weights + i * encodingSize;

        for (int e = 0; e < encodingSize; ++e) {
            encodedInput[e] += inputValue*w[e];
        }
    }

    // Sine / Cosine alternation:
    const float halvedOutputSize = encodingSize/2;
    for (int eSine = 0; eSine < std::ceil(halvedOutputSize); ++eSine) {
        double denom = pow(10000.0, (2.0 * eSine) / encodingSize);
        double angle = pos / denom;

        encodedInput[eSine*2] = sin(angle);
    }
    for (int eCosine = 0; eCosine < std::floor(halvedOutputSize); ++eCosine) {
        double denom = pow(10000.0, (2.0 * eCosine) / encodingSize);
        double angle = pos / denom;

        encodedInput[eCosine*2 + 1] = cos(angle);
    }
}
// used to generate keys, values and queries
inline void generate_hidden(const neurologicalValue* __restrict encoding, neurologicalValue* __restrict hidden
    , const neurologicalValue* __restrict weights
    , const int& encodingSize)
{
    for (int i = 0; i < encodingSize; ++i) {
        const neurologicalValue& inputValue = encoding[i];

        const neurologicalValue* __restrict w = weights + i * encodingSize;

        for (int o = 0; o < encodingSize; ++o) {
            hidden[o] += inputValue*w[o];
        }
    }
}
// masked is as in "previous including this one"
inline void masked_attention(const neurologicalValue* __restrict maskedKeys
    , const neurologicalValue* __restrict maskedValues
    , const neurologicalValue* __restrict query
    , const int& maskedTokensAmount
    , const int& encodingSize
    
    , neurologicalValue* __restrict attentionValues_buffer
    , neurologicalValue* __restrict attention_buffer
    , neurologicalValue* __restrict masked_attention_output)
{
    for (int i = 0; i < maskedTokensAmount; ++i) {
        const neurologicalValue* __restrict key = maskedKeys + i * encodingSize;
        for (int e = 0; e < encodingSize; ++e) {
            attentionValues_buffer[i] += query[e]*key[e];
        }
    }
    softmax(attentionValues_buffer, attention_buffer, maskedTokensAmount);
    for (int i = 0; i < maskedTokensAmount; ++i) {
        const neurologicalValue* __restrict value = maskedValues + i * encodingSize;
        const neurologicalValue& attentionValue = attention_buffer[i];
        for (int e = 0; e < encodingSize; ++e) {
            masked_attention_output[e] += attentionValue*maskedValues[e];
        }
    }
}
inline void process_fully_connected(const neurologicalValue* __restrict encodedInput, const neurologicalValue* __restrict attentionOutput
    , neurologicalValue* __restrict output
    , const neurologicalValue* __restrict weights
    , const neurologicalValue* __restrict biases
    , const int& encodingSize, const int& outputSize)
{
    for (int e = 0; e < encodingSize; ++e) {
        const neurologicalValue residualConnection = encodedInput[e] + attentionOutput[e];

        const neurologicalValue* __restrict w = weights + e * outputSize;

        for (int o = 0; o < outputSize; ++o) {
            output[o] += residualConnection*w[o];
        }
    }
    for (int o = 0; o < outputSize; ++o) {
        output[o] += biases[o];
    }
}

// true functions
void hidden_masked_self_attention_layer(const neurologicalValue* __restrict encoding, neurologicalValue* __restrict hidden_output
    , const neurologicalValue* __restrict weights
    , const neurologicalValue* __restrict biases
    , neurologicalValue* __restrict keys
    , neurologicalValue* __restrict values
    , neurologicalValue* __restrict queries
    , const int& sampleIndex
    , const int& encodingSize, const int& resultSize
    
    , neurologicalValue* __restrict attentionValues_buffer
    , neurologicalValue* __restrict attention_buffer
    , neurologicalValue* __restrict masked_attention_output)
{
    neurologicalValue* __restrict targetKey = keys + sampleIndex * encodingSize;
    neurologicalValue* __restrict targetValue = keys + sampleIndex * encodingSize;
    neurologicalValue* __restrict targetQuery = keys + sampleIndex * encodingSize;

    const int& attentionWeightsSpace = encodingSize*encodingSize;

    // generate keys
    generate_hidden(encoding, targetKey, weights, encodingSize);
    weights += attentionWeightsSpace;
    // generate values
    generate_hidden(encoding, targetValue, weights, encodingSize);
    weights += attentionWeightsSpace;
    // generate queries
    generate_hidden(encoding, targetQuery, weights, encodingSize);
    weights += attentionWeightsSpace;

    masked_attention(keys, values, targetQuery, sampleIndex+1, encodingSize, attentionValues_buffer, attention_buffer, masked_attention_output);

    process_fully_connected(encoding, masked_attention_output, hidden_output, weights, biases, encodingSize, resultSize);
}

void transform(const neurologicalValue* __restrict input, neurologicalValue* __restrict output
    , const neurologicalValue* __restrict weights
    , const neurologicalValue* __restrict biases
    , neurologicalValue* __restrict encodedInput
    , neurologicalValue* __restrict keys
    , neurologicalValue* __restrict values
    , neurologicalValue* __restrict queries
    , const int& seriesLength
    , const int& inputSize, const int& outputSize, const int& encodingSize

    , neurologicalValue* __restrict hidden_output
    , neurologicalValue* __restrict attentionValues_buffer
    , neurologicalValue* __restrict attention_buffer
    , neurologicalValue* __restrict masked_attention_output)
{
    // WEIGHTS AND BIASES ARE EQUALLY SHARED BETWEEN SAMPLES! DO NOT "advance" them, only add on
    // them with respect to their usage

    for (int s = 0; s < seriesLength; ++s) {
        // encode input
        encode_input(input, encodedInput, weights, inputSize, encodingSize, s);
        // prcoess hidden layers (for now its just 1)
        hidden_masked_self_attention_layer(encodedInput, hidden_output, weights + inputSize*encodingSize, biases
            , keys, values, queries, s, encodingSize, outputSize
            , attentionValues_buffer, attention_buffer, masked_attention_output);

        // advance attention buffers
        hidden_output += outputSize;

        attentionValues_buffer += s+1;
        attention_buffer += s+1;

        masked_attention_output += encodingSize;

        softmax(hidden_output, output, outputSize);

        // advance buffers
        input += inputSize;
 
        encodedInput += encodingSize;

        output += outputSize;
    }
}