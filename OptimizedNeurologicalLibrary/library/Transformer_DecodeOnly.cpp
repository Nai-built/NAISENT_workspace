#include <iostream>

#include "Transformer_DecodeOnly.h"

constexpr int ATTENTION_GATES = 4;

constexpr int MAX_SERIES = 100000;

Transformer_DecodeOnly::Transformer_DecodeOnly(int _inputTokens, int _outputTokens, int _encodingSize) {
    this->componentType = NeurologicalComponentTypes::DECODER_ONLY_TRANSFORMER;

    this->inputTokens = _inputTokens;
    this->outputTokens = _outputTokens;
    this->encodingSize = _encodingSize;
}

void Transformer_DecodeOnly::Initialize(Transformer_DecodeOnly* transformer) {
    transformer->encodeWeights = neurologicalBuffer(transformer->inputTokens*transformer->encodingSize);
    transformer->attentionWeights =
        neurologicalBuffer(transformer->stacks.size()*transformer->encodingSize*transformer->encodingSize*4);

    transformer->decodeWeights = neurologicalBuffer(transformer->encodingSize*transformer->outputTokens);
    transformer->decodeBiases = neurologicalBuffer(transformer->outputTokens);
    
	neurologicalValue weights_initialization_boundary =
    (sqrt(2.0 / ((double)(transformer->encodingSize)))*sqrt(3));

    for (int i = 0; i < transformer->encodeWeights.size(); i++) {
        transformer->encodeWeights[i]
        = getRandom(-weights_initialization_boundary, weights_initialization_boundary);
    }
    for (int i = 0; i < transformer->attentionWeights.size(); i++) {
        transformer->attentionWeights[i]
        = getRandom(-weights_initialization_boundary, weights_initialization_boundary);
    }
    for (int i = 0; i < transformer->decodeWeights.size(); i++) {
        transformer->decodeWeights[i]
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

// multi head attention is done!
inline void multi_head_masked_self_attention(const neurologicalValue* __restrict encoding
    , const neurologicalValue* __restrict weights
    , const neurologicalValue* __restrict biases
    , neurologicalValue* __restrict keys_buffer
    , neurologicalValue* __restrict queries_buffer
    , neurologicalValue* __restrict values_buffer
    , const int& seriesLength
    , const int& encodingSize, const int& headEncodingSize
    
    // , neurologicalValue* __restrict attentionOdds_buffer

    , neurologicalValue* __restrict MAX_scores_buffer
    , neurologicalValue* __restrict heads_outputs
    , neurologicalValue* __restrict output

    , const int& heads)
{
    const neurologicalValue attentionScoreAlpha = 1/sqrt(headEncodingSize);

    for (int h = 0; h < heads; ++h) {
        // process keys, values and queries
        const neurologicalValue* __restrict headWeights = weights + h*encodingSize*headEncodingSize*3;

        // neurologicalValue* __restrict attentionOdds_headBuffer = attentionOdds_buffer + h*seriesLength*seriesLength;

        neurologicalValue* __restrict headKeysBuffer = keys_buffer + h*seriesLength*headEncodingSize;
        neurologicalValue* __restrict headQueriesBuffer = queries_buffer + h*seriesLength*headEncodingSize;
        neurologicalValue* __restrict headValuesBuffer = values_buffer + h*seriesLength*headEncodingSize;

        for (int s = 0; s < seriesLength; ++s) {
            neurologicalValue* __restrict headOutput = heads_outputs + s*heads*headEncodingSize
                + h*headEncodingSize;

            const neurologicalValue* __restrict encodingSample = encoding + s*encodingSize;

            for (int e = 0; e < encodingSize; ++e) {
                const neurologicalValue& encodingValue = encodingSample[e];

                const neurologicalValue* __restrict w = headWeights + e*headEncodingSize*3;

                for (int o = 0; o < headEncodingSize; ++o) {
                    headKeysBuffer[o] += encodingValue*w[0]; // key
                    headQueriesBuffer[o] += encodingValue*w[1]; // query
                    headValuesBuffer[o] += encodingValue*w[2]; // value
                    w+=3;
                }
            }

            // masked self attention process
            neurologicalValue expSum = 0.0;
            neurologicalValue* __restrict headTargetKeysBuffer = keys_buffer + h*seriesLength*headEncodingSize;

            for (int t = 0; t <= s; ++t) {
                neurologicalValue attentionScore = 0.0;
                for (int o = 0; o < headEncodingSize; ++o) {
                    attentionScore += headQueriesBuffer[o]*headTargetKeysBuffer[o];
                }
                const neurologicalValue v = std::exp(attentionScore*attentionScoreAlpha);
                MAX_scores_buffer[t] = v;
                expSum += v;

                headTargetKeysBuffer += headEncodingSize;
            }
            neurologicalValue* __restrict headTargetValuesBuffer = values_buffer + h*seriesLength*headEncodingSize;

            const neurologicalValue alpha = 1/expSum;

            for (int t = 0; t <= s; ++t) {
                const neurologicalValue influence = (MAX_scores_buffer[t]*alpha);
                for (int o = 0; o < headEncodingSize; ++o) {
                    headOutput[o] += headTargetValuesBuffer[o]*influence;
                }

                // attentionOdds_headBuffer[t] = influence;

                headTargetValuesBuffer += headEncodingSize;
            }
            
            // advance buffers
            // attentionOdds_headBuffer += s+1;

            headKeysBuffer+=headEncodingSize;
            headQueriesBuffer+=headEncodingSize;
            headValuesBuffer+=headEncodingSize;
        }
    }

    // final fully connected layer process to "mix" the heads results
    weights += heads*encodingSize*headEncodingSize*3;
    for (int s = 0; s < seriesLength; ++s) {
        const neurologicalValue* __restrict sampleAttentionOutput = heads_outputs + s*heads*headEncodingSize;

        for (int i = 0; i < encodingSize; ++i) {
            const neurologicalValue& attentionValue = sampleAttentionOutput[i];

            const neurologicalValue* __restrict w = weights + i * encodingSize;

            for (int o = 0; o < encodingSize; ++o) {
                output[o] += attentionValue*w[o];
            }
        }
        for (int o = 0; o < encodingSize; ++o) {
            output[o] += biases[o];
        }

        output += encodingSize;
    }
}

void encode_series(const neurologicalValue* __restrict input, neurologicalValue* __restrict encodedInput
    , const neurologicalValue* __restrict weights
    , const int& inputSize, const int& encodingSize
    , const int& seriesLength)
{
    for (int s = 0; s < seriesLength; ++s) {
        encode_input(input+s*inputSize, encodedInput+s*encodingSize, weights, inputSize, encodingSize, s);
    }
}
void process_stack(const TransformerStack& stack, const neurologicalValue* __restrict encoding
    , const neurologicalValue* __restrict attentionWeights
    , const neurologicalValue* __restrict attentionFinalBiases
    , neurologicalValue* __restrict keys_buffer
    , neurologicalValue* __restrict queries_buffer
    , neurologicalValue* __restrict values_buffer
    , const int& seriesLength
    , const int& encodingSize
    
    , neurologicalValue* __restrict MAX_scores_buffer
    , neurologicalValue* __restrict attention_heads_outputs
    , neurologicalValue* __restrict attention_output)
{
    const int headEncodingSize = encodingSize/stack.attentionHeads;

    multi_head_masked_self_attention(
        encoding,
        attentionWeights,
        attentionFinalBiases,

        keys_buffer, queries_buffer, values_buffer,

        seriesLength, encodingSize, headEncodingSize,

        MAX_scores_buffer,
        attention_heads_outputs,
        attention_output,

        stack.attentionHeads
    );

    
}

// this is enough for today

// next, redo the transformer, yes, again..
// this time however, make sure to start from the main functions so things are much clearer and easier to reason with

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

// also try weights decay cuz why not right?

// void multi_head_masked_self_attention(const neurologicalValue* __restrict encoding
//     , const neurologicalValue* __restrict weights
//     , const neurologicalValue* __restrict biases
//     , neurologicalValue* __restrict KQV_buffer
//     , const int& seriesLength
//     , const int& encodingSize, const int& headEncodingSize
    
//     , neurologicalValue* __restrict attentionScores_buffer
//     , neurologicalValue* __restrict attention_buffer

//     , neurologicalValue* __restrict attention_output
//     , neurologicalValue* __restrict output

//     , const int& heads)
// {
//     for (int h = 0; h < heads; ++h) {
//         // process keys, values and queries
//         neurologicalValue* __restrict headOutput = attention_output + h*headEncodingSize;

//         const neurologicalValue* __restrict headWeights = weights + h*encodingSize*headEncodingSize*3;

//         neurologicalValue* __restrict attnetionScores_headBuffer = attentionScores_buffer + h*seriesLength;
//         neurologicalValue* __restrict attnetion_headBuffer = attention_buffer + h*seriesLength;

//         for (int s = 0; s < seriesLength; ++s) {
//             neurologicalValue* __restrict headSampleBuffer = KQV_buffer + h*s*headEncodingSize*3;

//             const neurologicalValue* __restrict encodingSample = encoding + s*encodingSize;

//             for (int e = 0; e < encodingSize; ++e) {
//                 const neurologicalValue& encodingValue = encodingSample[e];

//                 const neurologicalValue* __restrict w = headWeights + e*headEncodingSize*3;

//                 for (int o = 0; o < headEncodingSize; ++o) {
//                     headSampleBuffer[o*3] += encodingValue*w[0]; // key
//                     headSampleBuffer[o*3 + 1] += encodingValue*w[1]; // query
//                     headSampleBuffer[o*3 + 2] += encodingValue*w[2]; // value
//                     w+=3;
//                 }
//             }

//             // masked self attention process
//             for (int t = 0; t <= s; ++t) {
//                 neurologicalValue* __restrict headTargetSampleBuffer = KQV_buffer + h*t*headEncodingSize*3;
//                 neurologicalValue attentionScore = 0.0;
//                 for (int o = 0; o < headEncodingSize; ++o) {
//                     attentionScore += headSampleBuffer[o+1]*headTargetSampleBuffer[o];
//                 }
//                 attnetionScores_headBuffer[t] = attentionScore;
//             }
//             softmax(attnetionScores_headBuffer, attnetion_headBuffer, s+1);
//             for (int t = 0; t <= s; ++t) {
//                 neurologicalValue* __restrict headTargetSampleBuffer = KQV_buffer + h*t*headEncodingSize*3;
//                 for (int o = 0; o < headEncodingSize; ++o) {
//                     headOutput[o] += headTargetSampleBuffer[o+3]*attnetion_headBuffer[t];
//                 }
//             }
//         }
//     }

//     // final fully connected layer process to "mix" the heads results
//     weights += heads*encodingSize*headEncodingSize*3;
//     for (int i = 0; i < encodingSize; ++i) {
//         const neurologicalValue& attentionValue = attention_output[i];

//         const neurologicalValue* __restrict w = weights + i * encodingSize;

//         for (int o = 0; o < encodingSize; ++o) {
//             output[o] += attentionValue*w[o];
//         }
//     }
//     for (int o = 0; o < encodingSize; ++o) {
//         output[o] += biases[o];
//     }
// }

// this is enough for today

// TO DO: finish the transformer process, make the storing shape for data like keys, values and queries be like [heads, tokens, encoding]
// NOTE: we'll have to make each attention layer function account for the entire series and not just one sample