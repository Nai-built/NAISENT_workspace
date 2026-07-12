#include <iostream>

#include "ActivationFunctions/ReLU.h"

#include "DecoderOnly_Transformer.h"

constexpr neurologicalValue LOWEST = -3.40282e+38;

DecoderOnly_Transformer::DecoderOnly_Transformer(int _inputTokens, int _outputTokens, int _encodingSize) {
    this->componentType = NeurologicalComponentTypes::DECODER_ONLY_TRANSFORMER;

    this->inputTokens = _inputTokens;
    this->outputTokens = _outputTokens;
    this->encodingSize = _encodingSize;
}

void DecoderOnly_Transformer::Initialize(DecoderOnly_Transformer* transformer) {
    transformer->encodeWeights = neurologicalBuffer(transformer->inputTokens*transformer->encodingSize);
    transformer->encodeWeightsGradient = neurologicalBuffer(transformer->encodeWeights.size());

    transformer->attentionWeights =
        neurologicalBuffer(transformer->stacks.size()*transformer->encodingSize*transformer->encodingSize*4);
    transformer->attentionWeightsGradient = neurologicalBuffer(transformer->attentionWeights.size());
    
    transformer->attentionFinalBiases =
        neurologicalBuffer(transformer->stacks.size()*transformer->encodingSize);
    transformer->attentionFinalBiasesGradient = neurologicalBuffer(transformer->attentionFinalBiases.size());
        
    // by 2 because normalization happens before either an attention layer or a feed forward network
    transformer->normalizationGammas =
        neurologicalBuffer(transformer->stacks.size()*transformer->encodingSize*2);
    transformer->normalizationGammasGradient = neurologicalBuffer(transformer->normalizationGammas.size());

    for (int i = 0; i < transformer->normalizationGammas.size(); i++) {
        transformer->normalizationGammas[i] = 1.0f;
    }

    int feedForwardNetworksWeightsAmount = 0;
    int feedForwardNetworksBiasesAmount = 0;

    int feedForwardNetworksBufferSize = 0;

    for (int i = 0; i < transformer->stacks.size(); i++) {
        int lastSize = transformer->encodingSize;
        for (int j = 0; j <= transformer->stacks[i].hidden_FFN.size(); j++) {
            if (j == transformer->stacks[i].hidden_FFN.size()) {
                feedForwardNetworksWeightsAmount += lastSize*transformer->encodingSize;
                feedForwardNetworksBiasesAmount += transformer->encodingSize;
            } else {
                const int hidden_FFN_size = transformer->stacks[i].hidden_FFN[j];
                feedForwardNetworksWeightsAmount += lastSize*hidden_FFN_size;
                feedForwardNetworksBiasesAmount += hidden_FFN_size;
                lastSize = hidden_FFN_size;

                feedForwardNetworksBufferSize += hidden_FFN_size;
            }
        }
    }

    transformer->feedForwardNetworksWeights = neurologicalBuffer(feedForwardNetworksWeightsAmount);
    transformer->feedForwardNetworksWeightsGradient = neurologicalBuffer(transformer->feedForwardNetworksWeights.size());
    transformer->feedForwardNetworksBiases = neurologicalBuffer(feedForwardNetworksBiasesAmount);
    transformer->feedForwardNetworksBiasesGradient = neurologicalBuffer(transformer->feedForwardNetworksBiases.size());

    transformer->ffns_buffer_size = feedForwardNetworksBufferSize;

    transformer->decodeWeights = neurologicalBuffer(transformer->encodingSize*transformer->outputTokens);
    transformer->decodeWeightsGradient = neurologicalBuffer(transformer->decodeWeights.size());
    transformer->decodeBiases = neurologicalBuffer(transformer->outputTokens);
    transformer->decodeBiasesGradient = neurologicalBuffer(transformer->decodeBiases.size());
    
    neurologicalValue weights_initialization_boundary = 1.0 / sqrtf(transformer->encodingSize);

	// neurologicalValue encoding_weights_initialization_boundary =
    // (neurologicalValue)sqrt(6 / ((double)(transformer->inputTokens) + (double)(transformer->encodingSize)));;
	// neurologicalValue decoding_weights_initialization_boundary =
    // (neurologicalValue)sqrt(6 / ((double)(transformer->encodingSize) + (double)(transformer->outputTokens)));;
    
	// neurologicalValue attention_final_weights_initialization_boundary =
    // (neurologicalValue)sqrt(6 / ((double)(transformer->encodingSize) + (double)(transformer->encodingSize)));;

    for (int i = 0; i < transformer->encodeWeights.size(); i++) {
        transformer->encodeWeights[i]
        = getRandom(-weights_initialization_boundary, weights_initialization_boundary);
        // = getRandom(-encoding_weights_initialization_boundary, encoding_weights_initialization_boundary);
    }

    int attention_w_index = 0;
    int ffn_w_index = 0;

    const int squared_encodingSize = transformer->encodingSize*transformer->encodingSize;
    for (int s = 0; s < transformer->stacks.size(); s++) {
        // const int headEncodingSize = transformer->encodingSize/transformer->stacks[s].attentionHeads;
        // neurologicalValue attention_head_weights_initialization_boundary =
        // (neurologicalValue)sqrt(6 / ((double)(transformer->encodingSize) + (double)(headEncodingSize)));;

        // memory weights [keys, queries and values]
        for (int i = 0; i < squared_encodingSize*3; i++) {
            transformer->attentionWeights[attention_w_index]
            = getRandom(-weights_initialization_boundary, weights_initialization_boundary);
            // = getRandom(-attention_head_weights_initialization_boundary, attention_head_weights_initialization_boundary);
            attention_w_index++;
        }
        // final linear layer weights
        for (int i = 0; i < squared_encodingSize; i++) {
            transformer->attentionWeights[attention_w_index]
            = getRandom(-weights_initialization_boundary, weights_initialization_boundary);
            // = getRandom(-attention_final_weights_initialization_boundary, attention_final_weights_initialization_boundary);
            attention_w_index++;
        }
        
        // int lastSize = transformer->encodingSize;

    //     for (int j = 0; j <= transformer->stacks[s].hidden_FFN.size(); j++) {
    //         int ffl_input_size = lastSize;
    //         int ffl_output_size = 0;

    //         if (j == transformer->stacks[s].hidden_FFN.size()) {
    //             ffl_output_size = transformer->encodingSize;
    //         } else {
    //             const int hidden_FFN_size = transformer->stacks[s].hidden_FFN[j];
    //             ffl_output_size = hidden_FFN_size;
    //         }

    //         // neurologicalValue ffl_weights_initialization_boundary =
    //         // (neurologicalValue)sqrt(6 / ((double)(ffl_input_size) + (double)(ffl_output_size)));;

    //         for (int i = 0; i < ffl_input_size*ffl_output_size; i++) {
    //             transformer->feedForwardNetworksWeights[ffn_w_index]
    //             = getRandom(-weights_initialization_boundary, weights_initialization_boundary);
    //             // = getRandom(-ffl_weights_initialization_boundary, ffl_weights_initialization_boundary);
    //             ffn_w_index++;
    //         }
    //     }
    }
    for (int i = 0; i < transformer->feedForwardNetworksWeights.size(); i++) {
        transformer->feedForwardNetworksWeights[i]
            = getRandom(-weights_initialization_boundary, weights_initialization_boundary);
        // = getRandom(-decoding_weights_initialization_boundary, decoding_weights_initialization_boundary);
    }

    for (int i = 0; i < transformer->decodeWeights.size(); i++) {
        transformer->decodeWeights[i]
            = getRandom(-weights_initialization_boundary, weights_initialization_boundary);
        // = getRandom(-decoding_weights_initialization_boundary, decoding_weights_initialization_boundary);
    }
    
    // Sine / Cosine alternation initialization:
    const float halvedOutputSize = transformer->encodingSize/2;
    transformer->sineAlternationLength = std::ceil(halvedOutputSize);
    transformer->cosineAlternationLength = std::floor(halvedOutputSize);

    transformer->positionalSineMultipliers = vector<double>(transformer->sineAlternationLength);
    transformer->positionalCosineMultipliers = vector<double>(transformer->cosineAlternationLength);

    for (int eSine = 0; eSine < transformer->sineAlternationLength; ++eSine) {
        double denom = pow(10000.0, (2.0 * eSine) / transformer->encodingSize);
        double angleMultiplier = 1 / denom;

        transformer->positionalSineMultipliers[eSine] = angleMultiplier;
    }
    for (int eCosine = 0; eCosine < transformer->cosineAlternationLength; ++eCosine) {
        double denom = pow(10000.0, (2.0 * eCosine) / transformer->encodingSize);
        double angleMultiplier = 1 / denom;

        transformer->positionalCosineMultipliers[eCosine] = angleMultiplier;
    }
}

// the transformer prcoess will be like the following:
//      ~~~ handle input ~~~
//  - encode all series elements with encoding weights and add positional encoding.
//  - place that input encoding as the initial values for the transformer output stream.
//
// # TRANSFORMER STACK [duplicatable for however long you need!] -> {
//      ~~~ process attention layer ~~~
//  - using our output stream, produce all memory representations: keys, queries and values for each attention head.
//  - calculate masked self attention for each attention head.
//  - pass through a linear layer (X*W + B) to "mix" all the attention head's split productions.
//  - add that final attention output to our transformer output stream (residual connection).
//  - apply a normalization layer.
//      ~~~ process sub feed forward network ~~~
//  - pass our output stream into a sub feed forward network.
//  - the network may be just a single linear layer that has both its input size and output size as the actual stream size.
//  - or, it can have multiple layers with activation functions.
//  - example: linear layer (input: stream size, output: stream size * 4) -> relu activation -> linear layer (input: stream size * 4, output: stream size).
//  - in either case, both the output size and the input size of the feed forward network MUST match the stream size.
//  - again, add the final output from that FFN to the transformer output stream (residual connection).
//  - apply a normalization layer.
// # }
//
//  - finally.. take the final transformer output stream, and put it into a single linear layer (X*W + B).
//  - the linear properties are (input: stream size, output: output tokens size).
//  - the output from that final layer is our final output for the transformer.
//  - no residual connection, no normalization.
//  - just apply softmax to change it from just an output to "probabilities for each available output token".
//
//  - and... thats it! thats all, i believe.

void encode_input(const neurologicalValue* __restrict input, neurologicalValue* __restrict encoding
    , const neurologicalValue* __restrict inputWeights
    , const int& batchSize, const int* __restrict lengths
    , const int& inputSize, const int& encodingSize
    
    , const double* __restrict positionalSineMultipliers, const double* __restrict positionalSCosineMultipliers
    , const int& sineLength, const int& cosineLength)
{
    for (int b = 0; b < batchSize; ++b) {
        const int& length = lengths[b];
        for (int pos = 0; pos < length; ++pos) {
            // fully connected process
            for (int i = 0; i < inputSize; ++i) {
                const neurologicalValue& inputValue = input[i];

                const neurologicalValue* __restrict w = inputWeights + i * encodingSize;

                for (int e = 0; e < encodingSize; ++e) {
                    encoding[e] += inputValue*w[e];
                }
            }
            // Sine / Cosine alternation:
            for (int eSine = 0; eSine < sineLength; ++eSine) {
                encoding[eSine*2] += sin(pos * positionalSineMultipliers[eSine]);
            }
            for (int eCosine = 0; eCosine < cosineLength; ++eCosine) {
                encoding[eCosine*2 + 1] += cos(pos * positionalSCosineMultipliers[eCosine]);
            }

            encoding += encodingSize;
            input += inputSize;
        }
    }
}

constexpr neurologicalValue RMSNORM_EPSILON = 1e-6f;
void normalize_encoding_RMSNorm(const neurologicalValue* __restrict encoding, neurologicalValue* __restrict normalization
    , const neurologicalValue* __restrict gamma
    , const int& batchSize, const int* __restrict lengths
    , const int& encodingSize)
{
    for (int b = 0; b < batchSize; ++b) {
        const int& length = lengths[b];
        for (int pos = 0; pos < length; ++pos) {
            neurologicalValue mean_square = 0.0f;

            for (int e = 0; e < encodingSize; ++e) {
                const neurologicalValue& encodingValue = encoding[e];
                mean_square += encodingValue*encodingValue;
            }

            mean_square /= encodingSize;
            
            const neurologicalValue rMultiplier = 1.0f/sqrtf(mean_square+RMSNORM_EPSILON);

            for (int e = 0; e < encodingSize; ++e) {
                normalization[e] = encoding[e]*rMultiplier * gamma[e];
            }

            encoding += encodingSize;
            normalization += encodingSize;
        }
    }
}

void multi_head_compute_memory_buffers(const neurologicalValue* __restrict encoding
    , const neurologicalValue* __restrict attentionWeights
    , const int& batchSize, const int* __restrict lengths
    , const int& encodingSize, const int& headEncodingSize
    , neurologicalValue* __restrict keys_buffer
    , neurologicalValue* __restrict queries_buffer
    , neurologicalValue* __restrict values_buffer

    , const int& headsAmount)
{
    for (int b = 0; b < batchSize; ++b) {
        const int& length = lengths[b];
        for (int h = 0; h < headsAmount; ++h) {
            const neurologicalValue* __restrict _encoding = encoding;
            for (int pos = 0; pos < length; ++pos) {
                const neurologicalValue* __restrict w = attentionWeights + h*encodingSize*headEncodingSize*3;

                for (int e = 0; e < encodingSize; ++e) {
                    const neurologicalValue& encodingValue = _encoding[e];

                    for (int o = 0; o < headEncodingSize; ++o) {
                        keys_buffer[o] += encodingValue*w[0]; // key
                        queries_buffer[o] += encodingValue*w[1]; // query
                        values_buffer[o] += encodingValue*w[2]; // value
                        w+=3;
                    }
                }

                _encoding += encodingSize;

                keys_buffer += headEncodingSize;
                queries_buffer += headEncodingSize;
                values_buffer += headEncodingSize;
            }
        }

        encoding += length*encodingSize;
    }
}

void multi_head_masked_self_attention(neurologicalValue* __restrict attention_encoding
    , const int& batchSize, const int* __restrict lengths
    , const int& encodingSize, const int& headEncodingSize
    , const neurologicalValue* __restrict keys_buffer
    , const neurologicalValue* __restrict queries_buffer
    , const neurologicalValue* __restrict values_buffer

    , neurologicalValue* __restrict attention_scores
    , neurologicalValue* __restrict attention_softmax

    , const int& headsAmount)
{
    neurologicalValue _scale = 1.0f/sqrtf((float)encodingSize);

    for (int b = 0; b < batchSize; ++b) {
        const int& length = lengths[b];
        for (int h = 0; h < headsAmount; ++h) {
            for (int pos = 0; pos < length; ++pos) {
                neurologicalValue* __restrict sample_attention_encoding = attention_encoding + pos*encodingSize + h*headEncodingSize;
                
                const neurologicalValue* __restrict masked_keys = keys_buffer;
                neurologicalValue maxScore = LOWEST;
                for (int masked = 0; masked <= pos; ++masked) {
                    neurologicalValue score = 0.0;
                    for (int e = 0; e < headEncodingSize; ++e) {
                        score += queries_buffer[e] * masked_keys[e];
                    }
                    const neurologicalValue scoreValue = score*_scale;
                    attention_scores[masked] = scoreValue;
                    maxScore = std::max(maxScore, scoreValue);

                    masked_keys += headEncodingSize;
                }
                neurologicalValue expSum = 0.0;
                for (int masked = 0; masked <= pos; ++masked) {
                    const neurologicalValue expScore = std::expf(attention_scores[masked]-maxScore);
                    attention_scores[masked] = expScore;
                    expSum += expScore;
                }

                const neurologicalValue sumAlpha = 1.0f/expSum;
                
                const neurologicalValue* __restrict masked_values = values_buffer;
                for (int masked = 0; masked <= pos; ++masked) {
                    const neurologicalValue influence = attention_scores[masked]*sumAlpha;
                    for (int e = 0; e < headEncodingSize; ++e) {
                        sample_attention_encoding[e] += masked_values[e] * influence;
                    }
                    attention_softmax[masked] = influence;

                    masked_values += headEncodingSize;
                }

                attention_scores += pos+1;
                attention_softmax += pos+1;

                queries_buffer += headEncodingSize;
            }

            keys_buffer += length*headEncodingSize;
            values_buffer += length*headEncodingSize;
        }
        attention_encoding += length*encodingSize;
    }
}

void fully_connected_layer_weights(const neurologicalValue* __restrict encodingInput
    , neurologicalValue* __restrict encodingOutput
    , const neurologicalValue* __restrict weights
    , const int& batchSize, const int* __restrict lengths
    , const int& encodingInputSize, const int& encodingOutputSize)
{
    for (int b = 0; b < batchSize; ++b) {
        const int& length = lengths[b];
        for (int pos = 0; pos < length; ++pos) {
            for (int i = 0; i < encodingInputSize; ++i) {
                const neurologicalValue& inputValue = encodingInput[i];

                const neurologicalValue* __restrict w = weights + i * encodingOutputSize;

                for (int e = 0; e < encodingOutputSize; ++e) {
                    encodingOutput[e] += inputValue*w[e];
                }
            }

            encodingOutput += encodingOutputSize;
            encodingInput += encodingInputSize;
        }
    }
}

void fully_connected_layer_linear_activation(neurologicalValue* __restrict encodingOutput
    , const neurologicalValue* __restrict biases
    , const int& batchSize, const int* __restrict lengths
    , const int& encodingOutputSize)
{
    for (int b = 0; b < batchSize; ++b) {
        const int& length = lengths[b];
        for (int pos = 0; pos < length; ++pos) {
            for (int e = 0; e < encodingOutputSize; ++e) {
                encodingOutput[e] += biases[e];
            }

            encodingOutput += encodingOutputSize;
        }
    }
}

inline neurologicalValue ReLU__result(const neurologicalValue& leakage, const neurologicalValue& v) {
    if (v < 0) {
        return v*leakage;
    }
    return v;
}
void fully_connected_layer_relu_activation(neurologicalValue* __restrict encodingOutput
    , const neurologicalValue* __restrict biases
    , const int& batchSize, const int* __restrict lengths
    , const int& encodingOutputSize
    , neurologicalValue* __restrict cached_encodingOutput, const neurologicalValue& leakage)
{
    for (int b = 0; b < batchSize; ++b) {
        const int& length = lengths[b];
        for (int pos = 0; pos < length; ++pos) {
            for (int e = 0; e < encodingOutputSize; ++e) {
                const neurologicalValue v = encodingOutput[e] + biases[e];
                cached_encodingOutput[e] = v;
                encodingOutput[e] = ReLU__result(leakage, v);
            }

            cached_encodingOutput += encodingOutputSize;
            encodingOutput += encodingOutputSize;
        }
    }
}

void fully_connected_layer_linear_gradient(const neurologicalValue* __restrict propagation
    , neurologicalValue* __restrict biasesGradient
    , const int& batchSize, const int* __restrict lengths
    , const int& encodingOutputSize)
{
    for (int b = 0; b < batchSize; ++b) {
        const int& length = lengths[b];
        for (int pos = 0; pos < length; ++pos) {
            for (int e = 0; e < encodingOutputSize; ++e) {
                biasesGradient[e] +=  propagation[e];
            }

            propagation += encodingOutputSize;
        }
    }
}

inline neurologicalValue ReLU__gradient(const neurologicalValue& leakage, const neurologicalValue& v, const neurologicalValue& g) {
    if (v < 0) {
        return g*leakage;
    }
    return g;
}
void fully_connected_layer_relu_gradient(neurologicalValue* __restrict propagation
    , neurologicalValue* __restrict biasesGradient
    , const int& batchSize, const int* __restrict lengths
    , const int& encodingOutputSize

    , const neurologicalValue* __restrict cached_output, const neurologicalValue& leakage)
{
    for (int b = 0; b < batchSize; ++b) {
        const int& length = lengths[b];
        for (int pos = 0; pos < length; ++pos) {
            for (int e = 0; e < encodingOutputSize; ++e) {
                const neurologicalValue g = ReLU__gradient(leakage, cached_output[e], propagation[e]);
                propagation[e] = g;
                biasesGradient[e] += g;
            }

            cached_output += encodingOutputSize;
            propagation += encodingOutputSize;
        }
    }
}

void fully_connected_layer_weights_gradient(const neurologicalValue* __restrict propagation
    , const neurologicalValue* __restrict cached_encodingInput
    , neurologicalValue* __restrict weightsGradient
    , const int& batchSize, const int* __restrict lengths
    , const int& encodingInputSize, const int& encodingOutputSize)
{
    for (int b = 0; b < batchSize; ++b) {
        const int& length = lengths[b];
        for (int pos = 0; pos < length; ++pos) {
            for (int i = 0; i < encodingInputSize; ++i) {
                const neurologicalValue& cachedX = cached_encodingInput[i];

                neurologicalValue* __restrict wG = weightsGradient + i * encodingOutputSize;

                for (int e = 0; e < encodingOutputSize; ++e) {
                    wG[e] += cachedX*propagation[e];
                }
            }

            propagation += encodingOutputSize;
            cached_encodingInput += encodingInputSize;
        }
    }
}

void fully_connected_layer_input_gradient(const neurologicalValue* __restrict propagation
    , neurologicalValue* __restrict encodingInputPropagation
    , const neurologicalValue* __restrict weights
    , const int& batchSize, const int* __restrict lengths
    , const int& encodingInputSize, const int& encodingOutputSize)
{
    for (int b = 0; b < batchSize; ++b) {
        const int& length = lengths[b];
        for (int pos = 0; pos < length; ++pos) {
            for (int i = 0; i < encodingInputSize; ++i) {
                neurologicalValue g = 0.0;

                const neurologicalValue* __restrict w = weights + i * encodingOutputSize;

                for (int e = 0; e < encodingOutputSize; ++e) {
                    g += w[e]*propagation[e];
                }

                encodingInputPropagation[i] += g;
            }

            propagation += encodingOutputSize;
            encodingInputPropagation += encodingInputSize;
        }
    }
}

void multi_head_masked_self_attention_gradient(const neurologicalValue* __restrict attention_propagation
    , const int& batchSize, const int* __restrict lengths
    , const int& encodingSize, const int& headEncodingSize
    , const neurologicalValue* __restrict keys_buffer
    , const neurologicalValue* __restrict queries_buffer
    , const neurologicalValue* __restrict values_buffer

    , const neurologicalValue* __restrict attention_scores
    , const neurologicalValue* __restrict attention_softmax

    , neurologicalValue* __restrict keys_buffer_propagation
    , neurologicalValue* __restrict queries_buffer_propagation
    , neurologicalValue* __restrict values_buffer_propagation

    , neurologicalValue* __restrict attention_scores_gradient
    , neurologicalValue* __restrict attention_softmax_gradient

    , const int& headsAmount)
{
    neurologicalValue _scale = 1.0f/sqrtf((float)encodingSize);

    for (int b = 0; b < batchSize; ++b) {
        const int& length = lengths[b];
        for (int h = 0; h < headsAmount; ++h) {
            for (int pos = 0; pos < length; ++pos) {
                const neurologicalValue* __restrict sample_attention_propagation = attention_propagation + pos*encodingSize + h*headEncodingSize;
                
                neurologicalValue* __restrict masked_values_propagation = values_buffer_propagation;
                const neurologicalValue* __restrict masked_values = values_buffer;
                for (int masked = 0; masked <= pos; ++masked) {
                    const neurologicalValue influence = attention_softmax[masked];
                    neurologicalValue softmaxDerivative = 0.0f;
                    for (int e = 0; e < headEncodingSize; ++e) {
                        const neurologicalValue& pValue = sample_attention_propagation[e];

                        masked_values_propagation[e] += pValue * influence;
                        softmaxDerivative += pValue * masked_values[e];
                    }

                    masked_values_propagation += headEncodingSize;
                    masked_values += headEncodingSize;

                    attention_softmax_gradient[masked] += softmaxDerivative;
                }

                neurologicalValue softmaxDot = 0.0f;
                for (int masked = 0; masked <= pos; ++masked) {
                    softmaxDot += attention_softmax[masked]*attention_softmax_gradient[masked];
                }
                for (int masked = 0; masked <= pos; ++masked) {
                    attention_scores_gradient[masked] += attention_softmax[masked]
                    *(attention_softmax_gradient[masked]-softmaxDot);
                }

                neurologicalValue* __restrict masked_keys_propagation = keys_buffer_propagation;
                const neurologicalValue* __restrict masked_keys = keys_buffer;
                for (int masked = 0; masked <= pos; ++masked) {
                    neurologicalValue score_derivative = attention_scores_gradient[masked];
                    for (int e = 0; e < headEncodingSize; ++e) {
                        queries_buffer_propagation[e] += score_derivative * masked_keys[e] * _scale;
                        masked_keys_propagation[e] += score_derivative * queries_buffer[e] * _scale;
                    }
                    
                    masked_keys_propagation += headEncodingSize;
                    masked_keys += headEncodingSize;
                }

                attention_scores += pos+1;
                attention_softmax += pos+1;
                attention_scores_gradient += pos+1;
                attention_softmax_gradient += pos+1;

                queries_buffer_propagation += headEncodingSize;
                queries_buffer += headEncodingSize;
            }

            keys_buffer_propagation += length*headEncodingSize;
            values_buffer_propagation += length*headEncodingSize;
            keys_buffer += length*headEncodingSize;
            values_buffer += length*headEncodingSize;
        }
        attention_propagation += length*encodingSize;
    }
}

void multi_head_input_gradient_memory_buffers(neurologicalValue* __restrict inputEncodingPropagation
    , const neurologicalValue* __restrict attentionWeights
    , const int& batchSize, const int* __restrict lengths
    , const int& encodingSize, const int& headEncodingSize
    , const neurologicalValue* __restrict keys_buffer_propagation
    , const neurologicalValue* __restrict queries_buffer_propagation
    , const neurologicalValue* __restrict values_buffer_propagation

    , const int& headsAmount)
{
    for (int b = 0; b < batchSize; ++b) {
        const int& length = lengths[b];
        for (int h = 0; h < headsAmount; ++h) {
            neurologicalValue* __restrict _inputEncodingPropagation = inputEncodingPropagation;
            for (int pos = 0; pos < length; ++pos) {
                const neurologicalValue* __restrict w = attentionWeights + h*encodingSize*headEncodingSize*3;

                for (int e = 0; e < encodingSize; ++e) {
                    neurologicalValue encodingInputPropagationValue = 0.0f;

                    for (int o = 0; o < headEncodingSize; ++o) {
                        // if (keys_buffer_propagation[o] == 0.0f
                        //     && queries_buffer_propagation[o] == 0.0f
                        //     && values_buffer_propagation[o] == 0.0f) {
                        //     std::cout << "NOO" << std::endl;
                        // }
                        encodingInputPropagationValue += keys_buffer_propagation[o]*w[0]; // key
                        encodingInputPropagationValue += queries_buffer_propagation[o]*w[1]; // query
                        encodingInputPropagationValue += values_buffer_propagation[o]*w[2]; // value
                        w+=3;
                    }

                    _inputEncodingPropagation[e] += encodingInputPropagationValue;
                }

                _inputEncodingPropagation += encodingSize;

                keys_buffer_propagation += headEncodingSize;
                queries_buffer_propagation += headEncodingSize;
                values_buffer_propagation += headEncodingSize;
            }
        }

        inputEncodingPropagation += length*encodingSize;
    }
}
void multi_head_weights_gradient_memory_buffers(const neurologicalValue* __restrict cachedEncoding
    , neurologicalValue* __restrict attentionWeightsGradient
    , const int& batchSize, const int* __restrict lengths
    , const int& encodingSize, const int& headEncodingSize
    , const neurologicalValue* __restrict keys_buffer_propagation
    , const neurologicalValue* __restrict queries_buffer_propagation
    , const neurologicalValue* __restrict values_buffer_propagation

    , const int& headsAmount)
{
    for (int b = 0; b < batchSize; ++b) {
        const int& length = lengths[b];
        for (int h = 0; h < headsAmount; ++h) {
            const neurologicalValue* __restrict _cachedEncoding = cachedEncoding;
            for (int pos = 0; pos < length; ++pos) {
                neurologicalValue* __restrict wG = attentionWeightsGradient + h*encodingSize*headEncodingSize*3;

                for (int e = 0; e < encodingSize; ++e) {
                    const neurologicalValue& cachedEncodingValue = _cachedEncoding[e];

                    for (int o = 0; o < headEncodingSize; ++o) {
                        wG[0] += keys_buffer_propagation[o]*cachedEncodingValue; // key
                        wG[1] += queries_buffer_propagation[o]*cachedEncodingValue; // query
                        wG[2] += values_buffer_propagation[o]*cachedEncodingValue; // value
                        wG+=3;
                    }
                }

                _cachedEncoding += encodingSize;

                keys_buffer_propagation += headEncodingSize;
                queries_buffer_propagation += headEncodingSize;
                values_buffer_propagation += headEncodingSize;
            }
        }

        cachedEncoding += length*encodingSize;
    }
}

void normalize_encoding_RMSNorm_gradient(const neurologicalValue* __restrict propagation, neurologicalValue* __restrict encodingInputPropagation
    , const neurologicalValue* __restrict cached_encodingInput
    , const neurologicalValue* __restrict gamma, neurologicalValue* __restrict gammaGradient
    , const int& batchSize, const int* __restrict lengths
    , const int& encodingSize)
{
    for (int b = 0; b < batchSize; ++b) {
        const int& length = lengths[b];
        for (int pos = 0; pos < length; ++pos) {
            neurologicalValue mean_square = 0.0f;

            for (int e = 0; e < encodingSize; ++e) {
                const neurologicalValue& cached_encodingValue = cached_encodingInput[e];
                mean_square += cached_encodingValue*cached_encodingValue;
            }

            mean_square /= encodingSize;
            
            const neurologicalValue r = sqrtf(mean_square+RMSNORM_EPSILON);
            const neurologicalValue cubedR_by_sizeMultiplier = 1.0f/(encodingSize*r*r*r);
            const neurologicalValue rMultiplier = 1.0f/r;

            neurologicalValue dot = 0.0f;

            for (int e = 0; e < encodingSize; ++e)
                dot += propagation[e] * gamma[e] * cached_encodingInput[e];

            for (int e = 0; e < encodingSize; ++e) {
                const neurologicalValue& propagationValue = propagation[e];
                const neurologicalValue& cached_encodingValue = cached_encodingInput[e];

                float h = propagationValue * gamma[e];

                encodingInputPropagation[e] +=
                    (h * rMultiplier)
                    - (cached_encodingValue * dot) * cubedR_by_sizeMultiplier;

                // if (propagationValue == 0.0f) {
                //     std::cout << "NO P V = 0" << std::endl;
                // }
                if (cached_encodingValue == 0.0f) {
                    std::cout << "NO cached = 0" << std::endl;
                }
                if (rMultiplier == 0.0f) {
                    std::cout << "NO R = 0" << std::endl;
                }

                gammaGradient[e] += propagationValue * (cached_encodingValue * rMultiplier);
            }

            propagation += encodingSize;
            cached_encodingInput += encodingSize;
            encodingInputPropagation += encodingSize;
        }
    }
}

void ensureBuffer(neurologicalBuffer& buffer, const int& needed) {
    if (static_cast<size_t>(needed) > buffer.size()) {
        buffer.resize(needed);
    }
}

void DecoderOnly_Transformer::ForwardPass(DecoderOnly_Transformer* transformer, const neurologicalConstantSpan input, const neurologicalSpan output, const int& batchSize
    , const lengths seriesLengths, const int& totalSamples)
{
    std::cout << "total samples: " << totalSamples << std::endl;

    transformer->cached_input = input;

    // initialize ffns buffer
    zeroOutList(transformer->ffns_buffer);
    ensureBuffer(transformer->ffns_buffer, totalSamples*transformer->ffns_buffer_size);

    // initialize memory buffers
    const int needMemory_buffers_size = totalSamples * transformer->stacks.size()*transformer->encodingSize;
    zeroOutList(transformer->keys_buffer);
    ensureBuffer(transformer->keys_buffer, needMemory_buffers_size);
    zeroOutList(transformer->queries_buffer);
    ensureBuffer(transformer->queries_buffer, needMemory_buffers_size);
    zeroOutList(transformer->values_buffer);
    ensureBuffer(transformer->values_buffer, needMemory_buffers_size);

    // initialize stream encodings
    const int stream_buffer_size = totalSamples*transformer->encodingSize;
    zeroOutList(transformer->stream_encoding);
    ensureBuffer(transformer->stream_encoding, stream_buffer_size);

    // initialize stream encodings
    const int stream_buffer_stacks_size = stream_buffer_size*transformer->stacks.size();
    zeroOutList(transformer->stream_encoding_stacks);
    ensureBuffer(transformer->stream_encoding_stacks, stream_buffer_stacks_size*2);
    zeroOutList(transformer->stream_normalization_stacks);
    ensureBuffer(transformer->stream_normalization_stacks, stream_buffer_stacks_size*2);
    
    zeroOutList(transformer->stream_attention_encoding_stacks);
    ensureBuffer(transformer->stream_attention_encoding_stacks, stream_buffer_stacks_size);

    // initialize attention values
    int self_attention_amount = 0;
    for (int b = 0; b < batchSize; ++b) {
        const int& length = seriesLengths[b];
        self_attention_amount += length*length;
    }
    int totalAttentionHeadsAmount = 0;
    for (int s = 0; s < transformer->stacks.size(); ++s) {
        totalAttentionHeadsAmount += transformer->stacks[s].attentionHeads;
    }
    zeroOutList(transformer->attention_scores);
    ensureBuffer(transformer->attention_scores, self_attention_amount*totalAttentionHeadsAmount);
    zeroOutList(transformer->attention_softmax);
    ensureBuffer(transformer->attention_softmax, self_attention_amount*totalAttentionHeadsAmount);

    // encode input to stream encoding
    encode_input(
        input.data(),
        transformer->stream_encoding.data(),
        transformer->encodeWeights.data(),
        batchSize,
        seriesLengths.data(),
        transformer->inputTokens,
        transformer->encodingSize,

        transformer->positionalSineMultipliers.data(),
        transformer->positionalCosineMultipliers.data(),
        transformer->sineAlternationLength,
        transformer->cosineAlternationLength
    );
    
    // process each stack respectfully
    neurologicalValue* __restrict cache_stream_encoding = transformer->stream_encoding_stacks.data();
    neurologicalValue* __restrict stream_normalization = transformer->stream_normalization_stacks.data();
    neurologicalValue* __restrict stream_attention_encoding = transformer->stream_attention_encoding_stacks.data();

    const neurologicalValue* __restrict normalizationGamma = transformer->normalizationGammas.data();

    const neurologicalValue* __restrict attentionWeights = transformer->attentionWeights.data();
    const neurologicalValue* __restrict attentionFinalBiases = transformer->attentionFinalBiases.data();
    
    const neurologicalValue* __restrict FFNs_weights = transformer->feedForwardNetworksWeights.data();
    const neurologicalValue* __restrict FFNs_biases = transformer->feedForwardNetworksBiases.data();
    
    neurologicalValue* FFNs_buffer = transformer->ffns_buffer.data();

    neurologicalValue* __restrict keys_buffer = transformer->keys_buffer.data();
    neurologicalValue* __restrict queries_buffer = transformer->queries_buffer.data();
    neurologicalValue* __restrict values_buffer = transformer->values_buffer.data();
    
    neurologicalValue* __restrict attention_scores = transformer->attention_scores.data();
    neurologicalValue* __restrict attention_softmax = transformer->attention_softmax.data();
    
    for (int s = 0; s < transformer->stacks.size(); ++s) {
        // normalization layer process
        for (int i = 0; i < stream_buffer_size; ++i) {
            cache_stream_encoding[i] = transformer->stream_encoding[i];
        }
        cache_stream_encoding += stream_buffer_size;

        normalize_encoding_RMSNorm(
            transformer->stream_encoding.data(),
            stream_normalization,
            normalizationGamma,

            batchSize,
            seriesLengths.data(),
            transformer->encodingSize
        );
        normalizationGamma += transformer->encodingSize;

        const int headEncodingSize = transformer->encodingSize/transformer->stacks[s].attentionHeads;
        // generate attention memory buffers
        multi_head_compute_memory_buffers(
            stream_normalization, // input ~normalized
            attentionWeights,

            batchSize,
            seriesLengths.data(),
            transformer->encodingSize,
            headEncodingSize,

            keys_buffer,
            queries_buffer,
            values_buffer,

            transformer->stacks[s].attentionHeads
        );
        // masked self attention process
        multi_head_masked_self_attention(
            stream_attention_encoding,
            batchSize,
            seriesLengths.data(),
            transformer->encodingSize,
            headEncodingSize,

            keys_buffer,
            queries_buffer,
            values_buffer,

            attention_scores,
            attention_softmax,

            transformer->stacks[s].attentionHeads
        );

        keys_buffer += transformer->encodingSize*totalSamples;
        queries_buffer += transformer->encodingSize*totalSamples;
        values_buffer += transformer->encodingSize*totalSamples;

        attention_scores += self_attention_amount*transformer->stacks[s].attentionHeads;
        attention_softmax += self_attention_amount*transformer->stacks[s].attentionHeads;

        attentionWeights += transformer->encodingSize*transformer->encodingSize*3;
        
        // linear layer process [mixing heads]
        fully_connected_layer_weights(
            stream_attention_encoding, // input
            transformer->stream_encoding.data(), // output (+) residual connection
            attentionWeights,

            batchSize,
            seriesLengths.data(),
            transformer->encodingSize,
            transformer->encodingSize
        );
        attentionWeights += transformer->encodingSize*transformer->encodingSize;
        fully_connected_layer_linear_activation(
            transformer->stream_encoding.data(), // output (+) residual connection
            attentionFinalBiases,

            batchSize,
            seriesLengths.data(),
            transformer->encodingSize
        );
        attentionFinalBiases += transformer->encodingSize;

        stream_normalization += stream_buffer_size;

        // normalization layer process
        for (int i = 0; i < stream_buffer_size; ++i) {
            cache_stream_encoding[i] = transformer->stream_encoding[i];
        }
        cache_stream_encoding += stream_buffer_size;

        normalize_encoding_RMSNorm(
            transformer->stream_encoding.data(),
            stream_normalization,
            normalizationGamma,

            batchSize,
            seriesLengths.data(),
            transformer->encodingSize
        );
        normalizationGamma += transformer->encodingSize;

        // feed forward network
        int lastSize = transformer->encodingSize;

        neurologicalValue* ffl_input = stream_normalization;
        neurologicalValue* ffl_output = nullptr;

        for (int j = 0; j <= transformer->stacks[s].hidden_FFN.size(); ++j) {
            int ffl_input_size = lastSize;
            int ffl_output_size = 0;

            if (j == transformer->stacks[s].hidden_FFN.size()) {
                ffl_output_size = transformer->encodingSize;

                ffl_output = transformer->stream_encoding.data(); // finally add to the stream encoding (+) residual connection
            } else {
                const int hidden_FFN_size = transformer->stacks[s].hidden_FFN[j];
                ffl_output_size = hidden_FFN_size;

                ffl_output = FFNs_buffer;
            }

            const int ffl_buffer_size = totalSamples*ffl_output_size;

            fully_connected_layer_weights(
                ffl_input, // input
                ffl_output, // output

                FFNs_weights,

                batchSize,
                seriesLengths.data(),
                ffl_input_size,
                ffl_output_size
            );
            FFNs_weights += ffl_input_size*ffl_output_size;

            if (j == transformer->stacks[s].hidden_FFN.size()) {
                fully_connected_layer_linear_activation(
                    ffl_output, // output

                    FFNs_biases,

                    batchSize,
                    seriesLengths.data(),
                    ffl_output_size
                );
            }
            else {
                switch(transformer->stacks[s].activations_FFN[j]->activationType) {
                    case ActivationFunctionTypes::RELU: {
                        ReLU* relu = static_cast<ReLU*>(transformer->stacks[s].activations_FFN[j].get());
                        if (relu->cached_output.size() != ffl_buffer_size) {
                            relu->cached_output.resize(ffl_buffer_size);
                        }
                        fully_connected_layer_relu_activation(
                            ffl_output, // output

                            FFNs_biases,

                            batchSize,
                            seriesLengths.data(),
                            ffl_output_size,

                            relu->cached_output.data(),
                            relu->leakage
                        );
                        break;
                    }
                    default: {
                        fully_connected_layer_linear_activation(
                            ffl_output, // output

                            FFNs_biases,

                            batchSize,
                            seriesLengths.data(),
                            ffl_output_size
                        );
                        break;
                    }
                }
                
                ffl_input = FFNs_buffer;
                FFNs_buffer += ffl_buffer_size;
            }
            FFNs_biases += ffl_output_size;

            lastSize = ffl_output_size;
        }

        stream_normalization += stream_buffer_size;

        stream_attention_encoding += stream_buffer_size;
    }
    
    // decode output from stream encoding
    fully_connected_layer_weights(
        transformer->stream_encoding.data(), // input
        output.data(), // output

        transformer->decodeWeights.data(),

        batchSize,
        seriesLengths.data(),
        transformer->encodingSize,
        transformer->outputTokens
    );
    fully_connected_layer_linear_activation(
        output.data(), // output

        transformer->decodeBiases.data(),

        batchSize,
        seriesLengths.data(),
        transformer->outputTokens
    );
    // MAKE SURE TO APPLY SOFTMAX AFTER FORWARD PASSING!
}

// NEXT:
// - finish the transformer forward pass by adding the feed forward network process. [ DONE ] also initializatino is completely done!
// - add the transformer to the actual library, and test it through python to verify it works. [ DONE ] also fixed some issues with attention forward pass

// NEXT:
// - put together the back propagation process
// - test it in python and start training :>)

void update_parameter_list(neurologicalValue* __restrict parameters, const neurologicalValue* __restrict parametersGradient
    , const int& size
    , const neurologicalValue& rate)
{
    const neurologicalValue decayMultiplier = 1 - 0.001*rate;

    for (int i = 0; i < size; ++i)
    {
        parameters[i] = (parameters[i]*decayMultiplier) - (parametersGradient[i]*rate);
    }
}

void DecoderOnly_Transformer::BackPropagation(DecoderOnly_Transformer* transformer, const neurologicalSpan propagation, const neurologicalSpan inputPropagation, const neurologicalValue& learnRate, const int& batchSize
    , const lengths seriesLengths, const int& totalSamples)
{
    // initialize ffns buffer propagation
    zeroOutList(transformer->ffns_buffer_propagation);
    ensureBuffer(transformer->ffns_buffer_propagation, totalSamples*transformer->ffns_buffer_size);

    // initialize memory buffers propagation
    const int needMemory_buffers_size = totalSamples * transformer->stacks.size()*transformer->encodingSize;
    zeroOutList(transformer->keys_buffer_propagation);
    ensureBuffer(transformer->keys_buffer_propagation, needMemory_buffers_size);
    zeroOutList(transformer->queries_buffer_propagation);
    ensureBuffer(transformer->queries_buffer_propagation, needMemory_buffers_size);
    zeroOutList(transformer->values_buffer_propagation);
    ensureBuffer(transformer->values_buffer_propagation, needMemory_buffers_size);

    // initialize stream encodings propagation
    const int stream_buffer_size = totalSamples*transformer->encodingSize;
    zeroOutList(transformer->stream_encoding_propagation);
    ensureBuffer(transformer->stream_encoding_propagation, stream_buffer_size);

    // initialize stream encodings propagation
    const int stream_buffer_stacks_size = stream_buffer_size*transformer->stacks.size();
    zeroOutList(transformer->stream_normalization_stacks_propagation);
    ensureBuffer(transformer->stream_normalization_stacks_propagation, stream_buffer_stacks_size*2);
    
    zeroOutList(transformer->stream_attention_encoding_stacks_propagation);
    ensureBuffer(transformer->stream_attention_encoding_stacks_propagation, stream_buffer_stacks_size);

    // initialize attention values propagation
    int self_attention_amount = 0;
    for (int b = 0; b < batchSize; ++b) {
        const int& length = seriesLengths[b];
        self_attention_amount += length*length;
    }
    int totalAttentionHeadsAmount = 0;
    for (int s = 0; s < transformer->stacks.size(); ++s) {
        totalAttentionHeadsAmount += transformer->stacks[s].attentionHeads;
    }
    zeroOutList(transformer->attention_scores_tempGradient);
    ensureBuffer(transformer->attention_scores_tempGradient, self_attention_amount*totalAttentionHeadsAmount);
    zeroOutList(transformer->attention_softmax_tempGradient);
    ensureBuffer(transformer->attention_softmax_tempGradient, self_attention_amount*totalAttentionHeadsAmount);
    
    // std::cout << "final linear layer <<" << std::endl;

    fully_connected_layer_linear_gradient(
        propagation.data(),

        transformer->decodeBiasesGradient.data(),

        batchSize,
        seriesLengths.data(),
        transformer->outputTokens
    );

    fully_connected_layer_input_gradient(
        propagation.data(),
        transformer->stream_encoding_propagation.data(),

        transformer->decodeWeights.data(),

        batchSize,
        seriesLengths.data(),
        transformer->encodingSize,
        transformer->outputTokens
    );
    fully_connected_layer_weights_gradient(
        propagation.data(),
        transformer->stream_encoding.data(),
        
        transformer->decodeWeightsGradient.data(),

        batchSize,
        seriesLengths.data(),
        transformer->encodingSize,
        transformer->outputTokens
    );
    
    // parameters
    const neurologicalValue* __restrict normalizationGamma = transformer->normalizationGammas.data()
        + (transformer->normalizationGammas.size());

    const neurologicalValue* __restrict attentionWeights = transformer->attentionWeights.data()
        + (transformer->attentionWeights.size());
    const neurologicalValue* __restrict attentionFinalBiases = transformer->attentionFinalBiases.data()
        + (transformer->attentionFinalBiases.size());
    
    const neurologicalValue* __restrict FFNs_weights = transformer->feedForwardNetworksWeights.data()
        + (transformer->feedForwardNetworksWeights.size());
    const neurologicalValue* __restrict FFNs_biases = transformer->feedForwardNetworksBiases.data()
        + (transformer->feedForwardNetworksBiases.size());
    
    // parameters gradients
    neurologicalValue* __restrict normalizationGammaGradient = transformer->normalizationGammasGradient.data()
        + (transformer->normalizationGammasGradient.size());

    neurologicalValue* __restrict attentionWeightsGradient = transformer->attentionWeightsGradient.data()
        + (transformer->attentionWeightsGradient.size());
    neurologicalValue* __restrict attentionFinalBiasesGradient = transformer->attentionFinalBiasesGradient.data()
        + (transformer->attentionFinalBiasesGradient.size());
    
    neurologicalValue* __restrict FFNs_weightsGradient = transformer->feedForwardNetworksWeightsGradient.data()
        + (transformer->feedForwardNetworksWeightsGradient.size());
    neurologicalValue* __restrict FFNs_biasesGradient = transformer->feedForwardNetworksBiasesGradient.data()
        + (transformer->feedForwardNetworksBiasesGradient.size());

    // cached
    const neurologicalValue* __restrict cache_stream_encoding = transformer->stream_encoding_stacks.data()
        + (transformer->stream_encoding_stacks.size());
    const neurologicalValue* __restrict stream_normalization = transformer->stream_normalization_stacks.data()
        + (transformer->stream_normalization_stacks.size());
    const neurologicalValue* __restrict stream_attention_encoding = transformer->stream_attention_encoding_stacks.data()
        + (transformer->stream_attention_encoding_stacks.size());

    const neurologicalValue* FFNs_buffer = transformer->ffns_buffer.data()
        + (transformer->ffns_buffer.size());

    const neurologicalValue* __restrict keys_buffer = transformer->keys_buffer.data()
        + (transformer->keys_buffer.size());
    const neurologicalValue* __restrict queries_buffer = transformer->queries_buffer.data()
        + (transformer->queries_buffer.size());
    const neurologicalValue* __restrict values_buffer = transformer->values_buffer.data()
        + (transformer->values_buffer.size());
    
    const neurologicalValue* __restrict attention_scores = transformer->attention_scores.data()
        + (transformer->attention_scores.size());
    const neurologicalValue* __restrict attention_softmax = transformer->attention_softmax.data()
        + (transformer->attention_softmax.size());
    
    // propagation
    neurologicalValue* __restrict stream_normalization_propagation = transformer->stream_normalization_stacks_propagation.data()
        + (transformer->stream_normalization_stacks_propagation.size());
    neurologicalValue* __restrict stream_attention_encoding_propagation = transformer->stream_attention_encoding_stacks_propagation.data()
        + (transformer->stream_attention_encoding_stacks_propagation.size());

    neurologicalValue* FFNs_buffer_propagation = transformer->ffns_buffer_propagation.data()
        + (transformer->ffns_buffer_propagation.size());

    neurologicalValue* __restrict keys_buffer_propagation = transformer->keys_buffer_propagation.data()
        + (transformer->keys_buffer_propagation.size());
    neurologicalValue* __restrict queries_buffer_propagation = transformer->queries_buffer_propagation.data()
        + (transformer->queries_buffer_propagation.size());
    neurologicalValue* __restrict values_buffer_propagation = transformer->values_buffer_propagation.data()
        + (transformer->values_buffer_propagation.size());
    
    neurologicalValue* __restrict attention_scores_tempGradient = transformer->attention_scores_tempGradient.data()
        + (transformer->attention_scores_tempGradient.size());
    neurologicalValue* __restrict attention_softmax_tempGradient = transformer->attention_softmax_tempGradient.data()
        + (transformer->attention_softmax_tempGradient.size());
    
    // std::cout << "stacks <<" << std::endl;
    for (int s = transformer->stacks.size()-1; s >= 0; --s) {
        // std::cout << "feed forward netowrk <<" << std::endl;
        // feed forward network
        stream_normalization -= stream_buffer_size;
        stream_normalization_propagation -= stream_buffer_size;

        int lastSize = transformer->encodingSize;

        const neurologicalValue* ffl_cached_input = nullptr;
        neurologicalValue* ffl_inputPropagation = nullptr;
        neurologicalValue* ffl_propagation = nullptr;

        for (int j = transformer->stacks[s].hidden_FFN.size(); j >= 0; --j) {
            // std::cout << "feed forward layer << " << j << std::endl;

            int ffl_input_size = 0;
            int ffl_output_size = lastSize;

            if (j == transformer->stacks[s].hidden_FFN.size()) {
                ffl_propagation = transformer->stream_encoding_propagation.data(); // share propagation (+) residual connection
            } else {
                ffl_propagation = FFNs_buffer_propagation;
            }
            if (j == 0) {
                ffl_input_size = transformer->encodingSize;

                ffl_inputPropagation = stream_normalization_propagation;
                ffl_cached_input = stream_normalization;
            } else {
                const int previous_hidden_FFN_size = transformer->stacks[s].hidden_FFN[j-1];
                ffl_input_size = previous_hidden_FFN_size;

                const int ffl_buffer_size = totalSamples*ffl_input_size;

                ffl_inputPropagation = FFNs_buffer_propagation - ffl_buffer_size;
                ffl_cached_input = FFNs_buffer - ffl_buffer_size;

                FFNs_buffer_propagation -= ffl_buffer_size;
                FFNs_buffer -= ffl_buffer_size;
            }

            lastSize = ffl_input_size;

            FFNs_biasesGradient -= ffl_output_size;
            if (j == transformer->stacks[s].hidden_FFN.size()) {
                fully_connected_layer_linear_gradient(
                    ffl_propagation,

                    FFNs_biasesGradient,

                    batchSize,
                    seriesLengths.data(),
                    ffl_output_size
                );
            }
            else {
                switch(transformer->stacks[s].activations_FFN[j]->activationType) {
                    case ActivationFunctionTypes::RELU: {
                        ReLU* relu = static_cast<ReLU*>(transformer->stacks[s].activations_FFN[j].get());
                        fully_connected_layer_relu_gradient(
                            ffl_propagation,

                            FFNs_biasesGradient,

                            batchSize,
                            seriesLengths.data(),
                            ffl_output_size,

                            relu->cached_output.data(),
                            relu->leakage
                        );
                        break;
                    }
                    default: {
                        fully_connected_layer_linear_gradient(
                            ffl_propagation,

                            FFNs_biasesGradient,

                            batchSize,
                            seriesLengths.data(),
                            ffl_output_size
                        );
                        break;
                    }
                }
            }

            FFNs_weightsGradient -= ffl_input_size*ffl_output_size;
            FFNs_weights -= ffl_input_size*ffl_output_size;

            fully_connected_layer_input_gradient(
                ffl_propagation,
                ffl_inputPropagation,

                FFNs_weights,

                batchSize,
                seriesLengths.data(),
                ffl_input_size,
                ffl_output_size
            );
            fully_connected_layer_weights_gradient(
                ffl_propagation,
                ffl_cached_input,
                
                FFNs_weightsGradient,

                batchSize,
                seriesLengths.data(),
                ffl_input_size,
                ffl_output_size
            );
        }

        cache_stream_encoding -= stream_buffer_size;

        normalizationGamma -= transformer->encodingSize;
        normalizationGammaGradient -= transformer->encodingSize;
        normalize_encoding_RMSNorm_gradient(
            stream_normalization_propagation,
            transformer->stream_encoding_propagation.data(), // finally add to the stream encoding (+) residual connection
            cache_stream_encoding,
            normalizationGamma,
            normalizationGammaGradient,

            batchSize,
            seriesLengths.data(),
            transformer->encodingSize
        );

        // std::cout << "attention <<" << std::endl;
        // attention

        stream_normalization -= stream_buffer_size;
        stream_normalization_propagation -= stream_buffer_size;

        stream_attention_encoding -= stream_buffer_size;
        stream_attention_encoding_propagation -= stream_buffer_size;

        // linear layer process [mixing heads]
        // std::cout << "attention << biases" << std::endl;
        attentionFinalBiasesGradient -= transformer->encodingSize;
        fully_connected_layer_linear_gradient(
            transformer->stream_encoding_propagation.data(), // share propagation (+) residual connection

            attentionFinalBiasesGradient,

            batchSize,
            seriesLengths.data(),
            transformer->encodingSize
        );
        // std::cout << "attention << input" << std::endl;
        attentionWeights -= transformer->encodingSize*transformer->encodingSize;
        fully_connected_layer_input_gradient(
            transformer->stream_encoding_propagation.data(), // share propagation (+) residual connection
            stream_attention_encoding_propagation,

            attentionWeights,

            batchSize,
            seriesLengths.data(),
            transformer->encodingSize,
            transformer->encodingSize
        );
        // std::cout << "attention << weights" << std::endl;
        attentionWeightsGradient -= transformer->encodingSize*transformer->encodingSize;
        fully_connected_layer_weights_gradient(
            transformer->stream_encoding_propagation.data(), // share propagation (+) residual connection
            stream_attention_encoding,
            
            attentionWeightsGradient,

            batchSize,
            seriesLengths.data(),
            transformer->encodingSize,
            transformer->encodingSize
        );

        keys_buffer -= transformer->encodingSize*totalSamples;
        queries_buffer -= transformer->encodingSize*totalSamples;
        values_buffer -= transformer->encodingSize*totalSamples;

        attention_scores -= self_attention_amount*transformer->stacks[s].attentionHeads;
        attention_softmax -= self_attention_amount*transformer->stacks[s].attentionHeads;

        keys_buffer_propagation -= transformer->encodingSize*totalSamples;
        queries_buffer_propagation -= transformer->encodingSize*totalSamples;
        values_buffer_propagation -= transformer->encodingSize*totalSamples;

        attention_scores_tempGradient -= self_attention_amount*transformer->stacks[s].attentionHeads;
        attention_softmax_tempGradient -= self_attention_amount*transformer->stacks[s].attentionHeads;

        const int headEncodingSize = transformer->encodingSize/transformer->stacks[s].attentionHeads;
        // std::cout << "attention << masked self attention" << std::endl;
        multi_head_masked_self_attention_gradient(
            stream_attention_encoding_propagation,
            batchSize,
            seriesLengths.data(),
            transformer->encodingSize,
            headEncodingSize,

            keys_buffer,
            queries_buffer,
            values_buffer,

            attention_scores,
            attention_softmax,

            keys_buffer_propagation,
            queries_buffer_propagation,
            values_buffer_propagation,
            
            attention_scores_tempGradient,
            attention_softmax_tempGradient,

            transformer->stacks[s].attentionHeads
        );
        
        attentionWeights -= transformer->encodingSize*transformer->encodingSize*3;
        // std::cout << "attention << memory buffers" << std::endl;
        multi_head_input_gradient_memory_buffers(
            stream_normalization_propagation,
            attentionWeights,

            batchSize,
            seriesLengths.data(),
            transformer->encodingSize,
            headEncodingSize,

            keys_buffer_propagation,
            queries_buffer_propagation,
            values_buffer_propagation,
            
            transformer->stacks[s].attentionHeads
        );
        attentionWeightsGradient -= transformer->encodingSize*transformer->encodingSize*3;
        multi_head_weights_gradient_memory_buffers(
            stream_normalization,
            attentionWeightsGradient,

            batchSize,
            seriesLengths.data(),
            transformer->encodingSize,
            headEncodingSize,

            keys_buffer_propagation,
            queries_buffer_propagation,
            values_buffer_propagation,
            
            transformer->stacks[s].attentionHeads
        );

        cache_stream_encoding -= stream_buffer_size;

        normalizationGamma -= transformer->encodingSize;
        normalizationGammaGradient -= transformer->encodingSize;
        normalize_encoding_RMSNorm_gradient(
            stream_normalization_propagation,
            transformer->stream_encoding_propagation.data(), // finally add to the stream encoding (+) residual connection
            cache_stream_encoding,
            normalizationGamma,
            normalizationGammaGradient,

            batchSize,
            seriesLengths.data(),
            transformer->encodingSize
        );
    }

    // std::cout << "input encoding <<" << std::endl;
    
    fully_connected_layer_input_gradient(
        transformer->stream_encoding_propagation.data(),
        inputPropagation.data(),

        transformer->encodeWeights.data(),

        batchSize,
        seriesLengths.data(),
        transformer->inputTokens,
        transformer->encodingSize
    );
    fully_connected_layer_weights_gradient(
        transformer->stream_encoding_propagation.data(),
        transformer->cached_input.data(),
        
        transformer->encodeWeightsGradient.data(),

        batchSize,
        seriesLengths.data(),
        transformer->inputTokens,
        transformer->encodingSize
    );

    // updating
    // std::cout << "update <<" << std::endl;

    neurologicalValue rate = learnRate/(totalSamples/batchSize);

    update_parameter_list(
        transformer->encodeWeights.data(),
        transformer->encodeWeightsGradient.data(),
        transformer->encodeWeights.size(),
        rate
    );
    
    update_parameter_list(
        transformer->normalizationGammas.data(),
        transformer->normalizationGammasGradient.data(),
        transformer->normalizationGammas.size(),
        rate
    );

    update_parameter_list(
        transformer->attentionWeights.data(),
        transformer->attentionWeightsGradient.data(),
        transformer->attentionWeights.size(),
        rate
    );
    update_parameter_list(
        transformer->attentionFinalBiases.data(),
        transformer->attentionFinalBiasesGradient.data(),
        transformer->attentionFinalBiases.size(),
        rate
    );

    update_parameter_list(
        transformer->feedForwardNetworksWeights.data(),
        transformer->feedForwardNetworksWeightsGradient.data(),
        transformer->feedForwardNetworksWeights.size(),
        rate
    );
    update_parameter_list(
        transformer->feedForwardNetworksBiases.data(),
        transformer->feedForwardNetworksBiasesGradient.data(),
        transformer->feedForwardNetworksBiases.size(),
        rate
    );

    update_parameter_list(
        transformer->decodeWeights.data(),
        transformer->decodeWeightsGradient.data(),
        transformer->decodeWeights.size(),
        rate
    );
    update_parameter_list(
        transformer->decodeBiases.data(),
        transformer->decodeBiasesGradient.data(),
        transformer->decodeBiases.size(),
        rate
    );

    zeroOutList(transformer->encodeWeightsGradient);

    zeroOutList(transformer->normalizationGammasGradient);

    zeroOutList(transformer->attentionWeightsGradient);
    zeroOutList(transformer->attentionFinalBiasesGradient);
    
    zeroOutList(transformer->feedForwardNetworksWeightsGradient);
    zeroOutList(transformer->feedForwardNetworksBiasesGradient);
    
    zeroOutList(transformer->decodeWeightsGradient);
    zeroOutList(transformer->decodeBiasesGradient);
}