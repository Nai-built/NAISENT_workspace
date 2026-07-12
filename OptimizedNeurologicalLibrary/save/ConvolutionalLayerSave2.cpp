#include <iostream>

#include "ActivationFunctions/ReLU.h"

#include "ConvolutionalLayer.h"

#include "Extra.h"

constexpr neurologicalValue LOWEST = -3.40282e+38;
constexpr neurologicalValue LARGEST = 3.40282e+38;

ConvolutionalLayer::ConvolutionalLayer(int _inputChannels, int _outputChannels
        , int _kernelWidth, int _kernelHeight
        , int _inputChannelWidth, int _inputChannelHeight
        , int _stride, int _padding
        , ActivationFunction_UNIQUE _activationFunction
        , ChannelsPooler_UNIQUE _channelsPooler)
{
    this->componentType = NeurologicalComponentTypes::CONVOLUTIONAL_LAYER;

    this->inputChannels = _inputChannels;
    this->outputChannels = _outputChannels;
    
    this->kernelWidth = _kernelWidth;
    this->kernelHeight = _kernelHeight;
    
    this->inputChannelWidth = _inputChannelWidth;
    this->inputChannelHeight = _inputChannelHeight;
    
    this->stride = _stride;
    this->padding = _padding;

    if (_activationFunction == nullptr) {
        this->activationFunction = std::make_unique<IActivationFunction>();
        this->activationFunction->activationType = ActivationFunctionTypes::LINEAR;
    } else {
        this->activationFunction = std::move(_activationFunction);
    }
    if (_channelsPooler == nullptr) {
        this->channelsPooler = std::make_unique<IChannelsPooler>();
        this->channelsPooler->poolerType = ChannelsPoolerTypes::NONE;
    } else {
        this->channelsPooler = std::move(_channelsPooler);
    }
    
    this->outputChanelHeight = std::floor((this->inputChannelHeight-
        (this->kernelHeight)+(this->padding*2))/this->stride) + 1;

    this->outputChannelWidth = std::floor((this->inputChannelWidth-
        (this->kernelWidth)+(this->padding*2))/this->stride) + 1;

    this->inputSize = this->inputChannels*this->inputChannelWidth*this->inputChannelHeight;
    this->outputSize = this->outputChannels*this->outputChannelWidth*this->outputChanelHeight;
}

void ConvolutionalLayer::Initialize(ConvolutionalLayer* convolutionalLayer) {
    const int kernelSize = convolutionalLayer->kernelWidth * convolutionalLayer->kernelHeight;
    
    convolutionalLayer->weights = neurologicalBuffer(convolutionalLayer->inputChannels*kernelSize*convolutionalLayer->outputChannels);
    convolutionalLayer->biases = neurologicalBuffer(convolutionalLayer->outputChannels);

    convolutionalLayer->weightsGradient = neurologicalBuffer(convolutionalLayer->weights.size());
    convolutionalLayer->biasesGradient = neurologicalBuffer(convolutionalLayer->biases.size());

	neurologicalValue weights_initialization_boundary = (neurologicalValue)(sqrt(6 / ((double)(convolutionalLayer->inputChannels) + (double)(convolutionalLayer->outputChannels))));
	
    for (int i = 0; i < convolutionalLayer->outputChannels; i++) {
        convolutionalLayer->biases[i] = 0.0f;
        for (int j = 0; j < convolutionalLayer->inputChannels; j++) {
            // kernel
            for (int y = 0; y < convolutionalLayer->kernelHeight; y++) {
                for (int x = 0; x < convolutionalLayer->kernelWidth; x++) {
                    convolutionalLayer->weights
                    [j*kernelSize*convolutionalLayer->outputChannels
                        + y*convolutionalLayer->kernelWidth*convolutionalLayer->outputChannels
                        + x*convolutionalLayer->outputChannels
                        + i] =
                    getRandom(-weights_initialization_boundary, weights_initialization_boundary);
                }
            }
        }
    }
}
// after you find out how to peroply compute the weights, initialize them properly here [ DONE ]

constexpr int OUTPUT_BLOCK = 8;

// forward pass
void convolutional_weights_compute_OLD(const neurologicalValue* __restrict input, neurologicalValue* __restrict output, const neurologicalValue* __restrict weights
    , const int& batchSize
    , const int& inputChannels, const int& outputChannels
    , const int& kernelWidth, const int& kernelHeight
    , const int& inputChannelWidth, const int& inputChannelHeight
    , const int& outputChannelWidth, const int& outputChannelHeight)
{
    const int inputChannelSize = inputChannelWidth * inputChannelHeight;
    const int outputChannelSize = outputChannelWidth * outputChannelHeight;

    const int inputSampleSize = inputChannels * inputChannelSize;
    const int outputSampleSize = outputChannels * outputChannelSize;

    const int kernelSize = kernelWidth * kernelHeight;

    for (int b = 0; b < batchSize; b++)
    {
        // process sample

        neurologicalValue* __restrict y = output + b * outputSampleSize;
        const neurologicalValue* __restrict x = input + b * inputSampleSize;

        for (int o = 0; o < outputChannels; ++o) {
            neurologicalValue* __restrict yChannel = y + o * outputChannelSize;
            for (int i = 0; i < inputChannels; ++i) {
                const neurologicalValue* __restrict w = weights + (o * inputChannels * kernelSize) + (i * kernelSize);

                const neurologicalValue* __restrict xChannel = x + i * inputChannelSize;

                for (int oY = 0; oY < outputChannelHeight; ++oY) {
                    const int iCornerY = oY;
                    for (int oX = 0; oX < outputChannelWidth; ++oX) {
                        const int iCornerX = oX;

                        // kernel computation
                        for (int kY = 0; kY < kernelHeight; ++kY) {
                            for (int kX = 0; kX < kernelWidth; ++kX) {
                                yChannel[oY*outputChannelWidth + oX] += w[kY*kernelWidth + kX] * xChannel[(iCornerY+kY)*inputChannelWidth + (iCornerX+kX)];
                            }
                        }
                    }
                }
            }
        }
    }
}

// make the weights kernel more efficient and using its weights and input.. also dont forget to add stride and padding to it too. anyways, good work for now! [ DONE ]
void convolutional_weights_compute(const neurologicalValue* __restrict input, neurologicalValue* __restrict output, const neurologicalValue* __restrict weights
    , const int& batchSize
    , const int& inputChannels, const int& outputChannels
    , const int& kernelWidth, const int& kernelHeight
    , const int& inputChannelWidth, const int& inputChannelHeight
    , const int& outputChannelWidth, const int& outputChannelHeight
    , const int& stride, const int& padding)
{
    const int inputChannelSize = inputChannelWidth * inputChannelHeight;
    const int outputChannelSize = outputChannelWidth * outputChannelHeight;

    const int inputSampleSize = inputChannels * inputChannelSize;
    const int outputSampleSize = outputChannels * outputChannelSize;

    const int kernelSize = kernelWidth * kernelHeight;
    
    for (int b = 0; b < batchSize; b++)
    {
        // process sample

        neurologicalValue* __restrict y = output + b * outputSampleSize;
        const neurologicalValue* __restrict x = input + b * inputSampleSize;

        // output blocking
        for (int outputStart = 0; outputStart < outputChannels; outputStart+=OUTPUT_BLOCK) {
            const int oMax = std::min(outputStart + OUTPUT_BLOCK, outputChannels);

            const int blockSize = oMax - outputStart;

            // loop through the full output by point
            for (int oY = 0; oY < outputChannelHeight; ++oY) {
                const int iY = oY*stride - padding;
                for (int oX = 0; oX < outputChannelWidth; ++oX) {
                    const int iX = oX*stride - padding;

                    neurologicalValue accumulation[OUTPUT_BLOCK] = {0};
                    
                    const int xCorner = iY*inputChannelWidth + iX;

                    const int upClip = std::max(0, -iY);
                    const int downClip = std::max(0, (iY+kernelHeight)-inputChannelHeight);

                    const int leftClip = std::max(0, -iX);
                    const int rightClip = std::max(0, (iX+kernelWidth)-inputChannelWidth);

                    // accumulate for each output in input channels
                    for (int i = 0; i < inputChannels; ++i) {
                        const neurologicalValue* w_i = weights
                            + i*kernelSize*outputChannels;

                        const neurologicalValue* __restrict xChannel = x + i * inputChannelSize;

                        for (int kY = upClip; kY < kernelHeight-downClip; ++kY) {
                            const neurologicalValue* w_row = w_i
                                + kY*kernelWidth*outputChannels;

                            const int xRow = xCorner + kY*inputChannelWidth;

                            for (int kX = leftClip; kX < kernelWidth-rightClip; ++kX) {
                                const neurologicalValue xValue = xChannel[xRow + kX];

                                const neurologicalValue* __restrict w = w_row
                                + kX*outputChannels
                                + outputStart;

                                for (int o = 0; o < blockSize; ++o) {
                                    const neurologicalValue wValue = w[o];

                                    accumulation[o] += wValue*xValue;
                                }
                            }
                        }
                    }

                    // apply accumulation
                    for (int o = 0; o < blockSize; ++o) {
                        y[(o+outputStart)*outputChannelSize + oY*outputChannelWidth + oX] += accumulation[o];
                    }
                }
            }
        }
    }
}

// forward pass activations
void convolutional_activation_linear(neurologicalValue* __restrict output, const neurologicalValue* __restrict biases
    , const int& batchSize
    , const int& inputChannels, const int& outputChannels
    , const int& outputChannelWidth, const int& outputChannelHeight)
{
    const int outputChannelSize = outputChannelWidth * outputChannelHeight;
    
    const int outputSampleSize = outputChannels * outputChannelSize;

    for (int b = 0; b < batchSize; b++)
    {
        neurologicalValue* __restrict y = output + b * outputSampleSize;

        for (int o = 0; o < outputChannels; ++o) {
            const neurologicalValue& bias = biases[o];
            for (int oY = 0; oY < outputChannelHeight; ++oY) {
                for (int oX = 0; oX < outputChannelWidth; ++oX) {
                    y[o*outputChannelSize + oY*outputChannelWidth + oX] += bias;
                }
            }
        }
    }
}

inline neurologicalValue ReLU__result(const neurologicalValue& leakage, const neurologicalValue& v) {
    if (v < 0) {
        return v*leakage;
    }
    return v;
}
void convolutional_activation_relu(neurologicalValue* __restrict output, const neurologicalValue* __restrict biases
    , const int& batchSize
    , const int& inputChannels, const int& outputChannels
    , const int& outputChannelWidth, const int& outputChannelHeight
    , neurologicalValue* __restrict cached_output, const neurologicalValue& leakage)
{
    const int outputChannelSize = outputChannelWidth * outputChannelHeight;
    
    const int outputSampleSize = outputChannels * outputChannelSize;

    for (int b = 0; b < batchSize; b++)
    {
        neurologicalValue* __restrict y = output + b * outputSampleSize;
        neurologicalValue* __restrict cached_y = cached_output + b * outputSampleSize;

        for (int o = 0; o < outputChannels; ++o) {
            const neurologicalValue& bias = biases[o];
            for (int oY = 0; oY < outputChannelHeight; ++oY) {
                for (int oX = 0; oX < outputChannelWidth; ++oX) {
                    const int index = o*outputChannelSize + oY*outputChannelWidth + oX;

                    const neurologicalValue v = y[index] + bias;
                    cached_y[index] = v;
                    y[index] = ReLU__result(leakage, v);
                }
            }
        }
    }
}

void max_pool(const neurologicalValue* __restrict cached_output, neurologicalValue* __restrict output
    , const int& batchSize, const int& channelsAmount
    , const int& channelWidth, const int& channelHeight
    , const int& poolWidth, const int& poolHeight
    , const int& finalWidth, const int& finalHeight
    , const int& stride
    , int* __restrict picking_mask)
{
    int channelSize = channelWidth * channelHeight;
    int finalSize = finalWidth * finalHeight;

    int sampleSize = channelsAmount * channelSize;
    int finalSampleSize = channelsAmount * finalSize;

    int poolSize = poolWidth * poolHeight;

    for (int b = 0; b < batchSize; b++)
    {
        const neurologicalValue* __restrict cached_y = cached_output + b * sampleSize;
        neurologicalValue* __restrict final_y = output + b * finalSampleSize;

        for (int i = 0; i < channelsAmount; ++i) {
            // pool
            const neurologicalValue* __restrict yChannel = cached_y + i * channelSize;
            neurologicalValue* __restrict final_yChannel = final_y + i * channelSize;

            for (int y = 0; y < finalHeight; ++y) {
                const int cY = y*stride;

                for (int x = 0; x < finalWidth; ++x) {
                    const int cX = x*stride;

                    const int corner = cY*channelWidth + cX;

                    neurologicalValue poolValue = LOWEST;
                    int targetIndex = 0;

                    for (int pY = 0; pY < poolHeight; ++pY) {
                        const int row = corner + pY*channelWidth;
                        for (int pX = 0; pX < poolWidth; ++pX) {
                            const int xIndex = row + pX;
                            const neurologicalValue& xValue = yChannel[xIndex];
                            if (xValue > poolValue) {
                                poolValue = xValue;
                                targetIndex = xIndex;
                            }
                        }
                    }

                    final_yChannel[y*finalWidth + x] = poolValue;
                    picking_mask[y*finalWidth + x] = targetIndex;
                }
            }
        }
    }
}
void average_pool(const neurologicalValue* __restrict cached_output, neurologicalValue* __restrict output
    , const int& batchSize, const int& channelsAmount
    , const int& channelWidth, const int& channelHeight
    , const int& poolWidth, const int& poolHeight
    , const int& finalWidth, const int& finalHeight
    , const int& stride)
{
    int channelSize = channelWidth * channelHeight;
    int finalSize = finalWidth * finalHeight;

    int sampleSize = channelsAmount * channelSize;
    int finalSampleSize = channelsAmount * finalSize;

    int poolSize = poolWidth * poolHeight;

    for (int b = 0; b < batchSize; b++)
    {
        const neurologicalValue* __restrict cached_y = cached_output + b * sampleSize;
        neurologicalValue* __restrict final_y = output + b * finalSampleSize;

        for (int i = 0; i < channelsAmount; ++i) {
            // pool
            const neurologicalValue* __restrict yChannel = cached_y + i * channelSize;
            neurologicalValue* __restrict final_yChannel = final_y + i * channelSize;

            for (int y = 0; y < finalHeight; ++y) {
                const int cY = y*stride;

                for (int x = 0; x < finalWidth; ++x) {
                    const int cX = x*stride;

                    const int corner = cY*channelWidth + cX;

                    neurologicalValue poolValue = 0.0;

                    for (int pY = 0; pY < poolHeight; ++pY) {
                        const int row = corner + pY*channelWidth;
                        for (int pX = 0; pX < poolWidth; ++pX) {
                            poolValue += yChannel[row + pX];
                        }
                    }

                    final_yChannel[y*finalWidth + x] = poolValue/poolSize;
                }
            }
        }
    }
}
void min_pool(const neurologicalValue* __restrict cached_output, neurologicalValue* __restrict output
    , const int& batchSize, const int& channelsAmount
    , const int& channelWidth, const int& channelHeight
    , const int& poolWidth, const int& poolHeight
    , const int& finalWidth, const int& finalHeight
    , const int& stride
    , int* __restrict picking_mask)
{
    int channelSize = channelWidth * channelHeight;
    int finalSize = finalWidth * finalHeight;

    int sampleSize = channelsAmount * channelSize;
    int finalSampleSize = channelsAmount * finalSize;

    int poolSize = poolWidth * poolHeight;

    for (int b = 0; b < batchSize; b++)
    {
        const neurologicalValue* __restrict cached_y = cached_output + b * sampleSize;
        neurologicalValue* __restrict final_y = output + b * finalSampleSize;

        for (int i = 0; i < channelsAmount; ++i) {
            // pool
            const neurologicalValue* __restrict yChannel = cached_y + i * channelSize;
            neurologicalValue* __restrict final_yChannel = final_y + i * channelSize;

            for (int y = 0; y < finalHeight; ++y) {
                const int cY = y*stride;

                for (int x = 0; x < finalWidth; ++x) {
                    const int cX = x*stride;

                    const int corner = cY*channelWidth + cX;

                    neurologicalValue poolValue = LARGEST;
                    int targetIndex = 0;

                    for (int pY = 0; pY < poolHeight; ++pY) {
                        const int row = corner + pY*channelWidth;
                        for (int pX = 0; pX < poolWidth; ++pX) {
                            const int xIndex = row + pX;
                            const neurologicalValue& xValue = yChannel[xIndex];
                            if (xValue < poolValue) {
                                poolValue = xValue;
                                targetIndex = xIndex;
                            }
                        }
                    }

                    final_yChannel[y*finalWidth + x] = poolValue;
                    picking_mask[y*finalWidth + x] = targetIndex;
                }
            }
        }
    }
}

// back propagation activations
void convolutional_activation_gradient_linear(const neurologicalValue* __restrict propagation, neurologicalValue* __restrict biasesGradient
    , const int& batchSize
    , const int& inputChannels, const int& outputChannels
    , const int& outputChannelWidth, const int& outputChannelHeight)
{
    const int outputChannelSize = outputChannelWidth * outputChannelHeight;
    
    const int outputSampleSize = outputChannels * outputChannelSize;

    for (int b = 0; b < batchSize; b++)
    {
        const neurologicalValue* __restrict p = propagation + b * outputSampleSize;

        for (int o = 0; o < outputChannels; ++o) {
            neurologicalValue biasG = 0.0;
            for (int oY = 0; oY < outputChannelHeight; ++oY) {
                for (int oX = 0; oX < outputChannelWidth; ++oX) {
                    biasG += p[o*outputChannelSize + oY*outputChannelWidth + oX];
                }
            }
            biasesGradient[o] += biasG;
        }
    }
}

inline neurologicalValue ReLU__gradient(const neurologicalValue& leakage, const neurologicalValue& v, const neurologicalValue& g) {
    if (v < 0) {
        return g*leakage;
    }
    return g;
}
void convolutional_activation_gradient_relu(neurologicalValue* __restrict propagation, neurologicalValue* __restrict biasesGradient
    , const int& batchSize
    , const int& inputChannels, const int& outputChannels
    , const int& outputChannelWidth, const int& outputChannelHeight
    , const neurologicalValue* __restrict cached_output, const neurologicalValue& leakage)
{
    const int outputChannelSize = outputChannelWidth * outputChannelHeight;
    
    const int outputSampleSize = outputChannels * outputChannelSize;

    for (int b = 0; b < batchSize; b++)
    {
        const neurologicalValue* __restrict cached_y = cached_output + b * outputSampleSize;
        neurologicalValue* __restrict p = propagation + b * outputSampleSize;

        for (int o = 0; o < outputChannels; ++o) {
            neurologicalValue biasG = 0.0;
            for (int oY = 0; oY < outputChannelHeight; ++oY) {
                for (int oX = 0; oX < outputChannelWidth; ++oX) {
                    const int index = o*outputChannelSize + oY*outputChannelWidth + oX;

                    const neurologicalValue g = ReLU__gradient(leakage, cached_y[index], p[index]);
                    p[index] = g;
                    biasG += g;
                }
            }
            biasesGradient[o] += biasG;
        }
    }
}

// back propagation
neurologicalValue* packedPropagation = nullptr;
size_t packedCap = 0;

inline void ensurePacked(size_t needed)
{
    if (packedCap >= needed)
        return;

    if (packedPropagation)
        ::operator delete[](packedPropagation, std::align_val_t(64));

    packedPropagation = static_cast<float*>(
        ::operator new[](needed * sizeof(float), std::align_val_t(64))
    );

    packedCap = needed;
}
inline void pack(const neurologicalValue* __restrict propagation, neurologicalValue* __restrict packedPropagation
    , const int& outputChannelWidth, const int& outputChannelHeight, const int& outputChannelSize
    , const int& blockSize, const int& outputStart)
{
    for (int oY = 0; oY < outputChannelHeight; ++oY) {
        const int y = oY*blockSize*outputChannelWidth;
        for (int oX = 0; oX < outputChannelWidth; ++oX) {
            const int x = oX*blockSize;
            for (int o = 0; o < blockSize; o++) {
                packedPropagation[y + x + o] = propagation[(o+outputStart)*outputChannelSize + oY*outputChannelWidth + oX];
            }
        }
    }
}
void convolutional_propagation(const neurologicalValue* __restrict propagation
    , neurologicalValue* __restrict inputPropagation, const neurologicalValue* __restrict weights
    , const neurologicalValue* __restrict cached_input, neurologicalValue* __restrict weightsGradient
    , const int& batchSize
    , const int& inputChannels, const int& outputChannels
    , const int& kernelWidth, const int& kernelHeight
    , const int& inputChannelWidth, const int& inputChannelHeight
    , const int& outputChannelWidth, const int& outputChannelHeight
    , const int& stride, const int& padding)
{
    const int inputChannelSize = inputChannelWidth * inputChannelHeight;
    const int outputChannelSize = outputChannelWidth * outputChannelHeight;

    const int inputSampleSize = inputChannels * inputChannelSize;
    const int outputSampleSize = outputChannels * outputChannelSize;

    const int kernelSize = kernelWidth * kernelHeight;

    for (int b = 0; b < batchSize; b++)
    {
        const neurologicalValue* __restrict p = propagation + b * outputSampleSize;
        const neurologicalValue* __restrict cached_x = cached_input + b * inputSampleSize;

        neurologicalValue* __restrict inputP = inputPropagation + b * inputSampleSize;

        // output blocking
        for (int outputStart = 0; outputStart < outputChannels; outputStart+=OUTPUT_BLOCK) {
            const int oMax = std::min(outputStart + OUTPUT_BLOCK, outputChannels);

            const int blockSize = oMax - outputStart;

            ensurePacked(blockSize*outputChannelSize);
            pack(p, packedPropagation, outputChannelWidth, outputChannelHeight, outputChannelSize, blockSize, outputStart);

            // loop through the full propagation by point
            for (int oY = 0; oY < outputChannelHeight; ++oY) {
                const neurologicalValue* __restrict packedPropagationY = packedPropagation + oY*blockSize*outputChannelWidth;

                const int iY = oY*stride - padding;
                for (int oX = 0; oX < outputChannelWidth; ++oX) {
                    const neurologicalValue* __restrict packedPropagationX = packedPropagation + oX*blockSize;

                    const int iX = oX*stride - padding;

                    const int xCorner = iY*inputChannelWidth + iX;

                    const int upClip = std::max(0, -iY);
                    const int downClip = std::max(0, (iY+kernelHeight)-inputChannelHeight);

                    const int leftClip = std::max(0, -iX);
                    const int rightClip = std::max(0, (iX+kernelWidth)-inputChannelWidth);

                    // accumulate for each propagation in input channels
                    for (int i = 0; i < inputChannels; ++i) {
                        const neurologicalValue* w_i = weights
                            + i*kernelSize*outputChannels;

                        neurologicalValue* wG_i = weightsGradient
                            + i*kernelSize*outputChannels;

                        neurologicalValue* __restrict inputPropagationChannel = inputP + i * inputChannelSize;

                        const neurologicalValue* __restrict cached_xChannel = cached_x + i * inputChannelSize;

                        for (int kY = upClip; kY < kernelHeight-downClip; ++kY) {
                            const neurologicalValue* w_row = w_i
                                + kY*kernelWidth*outputChannels;

                            neurologicalValue* wG_row = wG_i
                                + kY*kernelWidth*outputChannels;

                            const int xRow = xCorner + kY*inputChannelWidth;

                            for (int kX = leftClip; kX < kernelWidth-rightClip; ++kX) {
                                const neurologicalValue cached_xValue = cached_xChannel[xRow + kX];
                                
                                neurologicalValue g = 0.0;

                                const neurologicalValue* __restrict w = w_row
                                + kX*outputChannels
                                + outputStart;

                                neurologicalValue* __restrict wG = wG_row
                                + kX*outputChannels
                                + outputStart;

                                for (int o = 0; o < blockSize; ++o) {
                                    const neurologicalValue wValue = w[o];

                                    g += wValue*packedPropagationX[o];

                                    wG[o] += cached_xValue*packedPropagationX[o];
                                }

                                inputPropagationChannel[xRow + kX] += g;
                            }
                        }
                    }
                }
            }
        }
    }
}

// updating
void update_weights_CONVO(neurologicalValue* __restrict weights, const neurologicalValue* __restrict weightsGradient
    , const int& gradientSize
    , const neurologicalValue& rate)
{
    for (int i = 0; i < gradientSize; ++i)
    {
        weights[i] -= weightsGradient[i] * rate;
    }
}
void update_biases_CONVO(neurologicalValue* __restrict biases, const neurologicalValue* __restrict biasesGradient
    , const int& gradientSize
    , const neurologicalValue& rate)
{
    for (int i = 0; i < gradientSize; ++i)
        biases[i] -= biasesGradient[i] * rate;
}

void ConvolutionalLayer::ForwardPass(ConvolutionalLayer* convolutionalLayer, const neurologicalConstantSpan input, const neurologicalSpan output) {    
    int batchSize = input.size()/convolutionalLayer->inputSize;
    
    convolutionalLayer->cached_input = input;

    convolutional_weights_compute(
        input.data(),
        output.data(),

        convolutionalLayer->weights.data(),
        
        batchSize,

        convolutionalLayer->inputChannels,
        convolutionalLayer->outputChannels,

        convolutionalLayer->kernelWidth,
        convolutionalLayer->kernelHeight,

        convolutionalLayer->inputChannelWidth,
        convolutionalLayer->inputChannelHeight,
        
        convolutionalLayer->outputChannelWidth,
        convolutionalLayer->outputChanelHeight,
        
        convolutionalLayer->stride,
        convolutionalLayer->padding
    );
    switch(convolutionalLayer->activationFunction->activationType) {
        case ActivationFunctionTypes::RELU: {
            ReLU* relu = static_cast<ReLU*>(convolutionalLayer->activationFunction.get());
            if (relu->cached_output.size() != output.size()) {
                relu->cached_output.resize(output.size());
            }
            convolutional_activation_relu(
                output.data(),

                convolutionalLayer->biases.data(),

                batchSize,
                    
                convolutionalLayer->inputChannels,
                convolutionalLayer->outputChannels,

                convolutionalLayer->outputChannelWidth,
                convolutionalLayer->outputChanelHeight,
            
                relu->cached_output.data(),
                relu->leakage
            );
            break;
        }
        default: {
            convolutional_activation_linear(
                output.data(),

                convolutionalLayer->biases.data(),

                batchSize,
                    
                convolutionalLayer->inputChannels,
                convolutionalLayer->outputChannels,

                convolutionalLayer->outputChannelWidth,
                convolutionalLayer->outputChanelHeight
            );
            break;
        }
    }
}

void ConvolutionalLayer::BackPropagation(ConvolutionalLayer* convolutionalLayer, const neurologicalSpan propagation, const neurologicalSpan inputPropagation, const neurologicalValue& learnRate) {
    int batchSize = propagation.size()/convolutionalLayer->outputSize;
    neurologicalValue rate = learnRate/batchSize;

    switch(convolutionalLayer->activationFunction->activationType) {
        case ActivationFunctionTypes::RELU: {
            ReLU* relu = static_cast<ReLU*>(convolutionalLayer->activationFunction.get());
            convolutional_activation_gradient_relu(
                propagation.data(),
                convolutionalLayer->biasesGradient.data(),

                batchSize,
                    
                convolutionalLayer->inputChannels,
                convolutionalLayer->outputChannels,

                convolutionalLayer->outputChannelWidth,
                convolutionalLayer->outputChanelHeight,
            
                relu->cached_output.data(),
                relu->leakage
            );
            break;
        }
        default: {
            convolutional_activation_gradient_linear(
                propagation.data(),
                convolutionalLayer->biasesGradient.data(),

                batchSize,
                    
                convolutionalLayer->inputChannels,
                convolutionalLayer->outputChannels,

                convolutionalLayer->outputChannelWidth,
                convolutionalLayer->outputChanelHeight
            );
            break;
        }
    }
    convolutional_propagation(
        propagation.data(),

        inputPropagation.data(),
        // unmodified weights
        convolutionalLayer->weights.data(),

        convolutionalLayer->cached_input.data(),
        convolutionalLayer->weightsGradient.data(),

        batchSize,

        convolutionalLayer->inputChannels,
        convolutionalLayer->outputChannels,

        convolutionalLayer->kernelWidth,
        convolutionalLayer->kernelHeight,

        convolutionalLayer->inputChannelWidth,
        convolutionalLayer->inputChannelHeight,
        
        convolutionalLayer->outputChannelWidth,
        convolutionalLayer->outputChanelHeight,
        
        convolutionalLayer->stride,
        convolutionalLayer->padding
    );

    update_biases_CONVO(
        convolutionalLayer->biases.data(),
        convolutionalLayer->biasesGradient.data(),

        convolutionalLayer->biases.size(),

        rate
    );
    update_weights_CONVO(
        convolutionalLayer->weights.data(),
        convolutionalLayer->weightsGradient.data(),

        convolutionalLayer->weights.size(),

        rate
    );

    zeroOutList(convolutionalLayer->weightsGradient);
    zeroOutList(convolutionalLayer->biasesGradient);

    // std::cout << "bias test: " << convolutionalLayer->biases[0] << std::endl;
    // std::cout << "weight test: " << convolutionalLayer->weights[0] << std::endl;
}