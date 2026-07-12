#include <iostream>

#include "ActivationFunctions/ReLU.h"

#include "ChannelsPoolers/Max.h"
#include "ChannelsPoolers/Average.h"
#include "ChannelsPoolers/Min.h"

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
    
    this->outputChannelHeight = std::floor((this->inputChannelHeight-
        (this->kernelHeight)+(this->padding*2))/this->stride) + 1;

    this->outputChannelWidth = std::floor((this->inputChannelWidth-
        (this->kernelWidth)+(this->padding*2))/this->stride) + 1;

    this->inputSize = this->inputChannels*this->inputChannelWidth*this->inputChannelHeight;
    this->outputSize = this->outputChannels*this->outputChannelWidth*this->outputChannelHeight;
    
    if (_channelsPooler == nullptr) {
        this->channelsPooler = std::make_unique<IChannelsPooler>();
        this->channelsPooler->poolerType = ChannelsPoolerTypes::NONE;

        this->finalChannelHeight = this->outputChannelHeight;
        this->finalChannelWidth = this->outputChannelWidth;
        this->finalOutputSize = this->outputSize;
    } else {
        this->channelsPooler = std::move(_channelsPooler);

        this->finalChannelHeight = std::floor((this->outputChannelHeight-
        (this->channelsPooler->poolingHeight))/this->channelsPooler->stride) + 1;

        this->finalChannelWidth = std::floor((this->outputChannelWidth-
        (this->channelsPooler->poolingWidth))/this->channelsPooler->stride) + 1;
        
        this->finalOutputSize = this->outputChannels*this->finalChannelWidth*this->finalChannelHeight;
    }
}

void ConvolutionalLayer::Initialize(ConvolutionalLayer* convolutionalLayer) {
    const int kernelSize = convolutionalLayer->kernelWidth * convolutionalLayer->kernelHeight;
    
    convolutionalLayer->weights = neurologicalBuffer(convolutionalLayer->inputChannels*kernelSize*convolutionalLayer->outputChannels);
    convolutionalLayer->biases = neurologicalBuffer(convolutionalLayer->outputChannels);

    convolutionalLayer->weightsGradient = neurologicalBuffer(convolutionalLayer->weights.size());
    convolutionalLayer->biasesGradient = neurologicalBuffer(convolutionalLayer->biases.size());

    // (neurologicalValue)(sqrt(6 / ((double)(convolutionalLayer->inputChannels) + (double)(convolutionalLayer->outputChannels))));

	neurologicalValue weights_initialization_boundary = (sqrt(2.0 / ((double)(convolutionalLayer->inputChannels*convolutionalLayer->kernelWidth*convolutionalLayer->kernelHeight)))*sqrt(3));
	
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

                const int upClip = std::max(0, -iY);
                const int downClip = std::max(0, (iY+kernelHeight)-inputChannelHeight);

                for (int oX = 0; oX < outputChannelWidth; ++oX) {
                    const int iX = oX*stride - padding;

                    neurologicalValue accumulation[OUTPUT_BLOCK] = {0};
                    
                    const int xCorner = iY*inputChannelWidth + iX;

                    const int leftClip = std::max(0, -iX);
                    const int rightClip = std::max(0, (iX+kernelWidth)-inputChannelWidth);

                    // accumulate for each output in input channels
                    for (int i = 0; i < inputChannels; ++i) {
                        // const neurologicalValue* w_i = weights
                        //     + i*kernelSize*outputChannels;

                        // const neurologicalValue* __restrict xCornerPointer = x + i * inputChannelSize + xCorner;

                        for (int kY = upClip; kY < kernelHeight-downClip; ++kY) {
                            const neurologicalValue* w_row = weights
                                + i*kernelSize*outputChannels

                                + kY*kernelWidth*outputChannels
                                + leftClip*outputChannels
                                + outputStart;

                            const neurologicalValue* __restrict xRow = x + i * inputChannelSize + xCorner
                            + kY*inputChannelWidth + leftClip;

                            for (int kX = leftClip; kX < kernelWidth-rightClip; ++kX) {
                                const neurologicalValue& xValue = *xRow;

                                for (int o = 0; o < blockSize; ++o) {
                                    const neurologicalValue& wValue = w_row[o];

                                    accumulation[o] += wValue*xValue;
                                }

                                xRow += 1;
                                w_row += outputChannels;
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
    , const int& outputChannels
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
    , const int& outputChannels
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

// pooling
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

    for (int b = 0; b < batchSize; b++)
    {
        const neurologicalValue* __restrict cached_y = cached_output + b * sampleSize;

        neurologicalValue* __restrict final_y = output + b * finalSampleSize;
        int* __restrict mask = picking_mask + b * finalSampleSize;

        for (int i = 0; i < channelsAmount; ++i) {
            // pool
            const neurologicalValue* __restrict yChannel = cached_y + i * channelSize;

            neurologicalValue* __restrict final_yChannel = final_y + i * finalSize;
            int* __restrict mask_channel = mask + i * finalSize;

            for (int y = 0; y < finalHeight; ++y) {
                const int cY = y*stride;

                const int upClip = std::max(0, -cY);
                const int downClip = std::max(0, (cY+poolHeight)-channelHeight);

                for (int x = 0; x < finalWidth; ++x) {
                    const int cX = x*stride;

                    const int corner = cY*channelWidth + cX;

                    const int leftClip = std::max(0, -cX);
                    const int rightClip = std::max(0, (cX+poolWidth)-channelWidth);

                    neurologicalValue poolValue = LOWEST;
                    int targetIndex = 0;

                    for (int pY = upClip; pY < poolHeight-downClip; ++pY) {
                        const int row = corner + pY*channelWidth;
                        for (int pX = leftClip; pX < poolWidth-rightClip; ++pX) {
                            const int yIndex = row + pX;
                            const neurologicalValue& yValue = yChannel[yIndex];
                            if (yValue > poolValue) {
                                poolValue = yValue;
                                targetIndex = yIndex;
                            }
                        }
                    }

                    // std::cout << "picked: " << y*finalWidth + x << " as " << targetIndex << std::endl;

                    final_yChannel[y*finalWidth + x] = poolValue;
                    mask_channel[y*finalWidth + x] = targetIndex;
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

    for (int b = 0; b < batchSize; b++)
    {
        const neurologicalValue* __restrict cached_y = cached_output + b * sampleSize;

        neurologicalValue* __restrict final_y = output + b * finalSampleSize;

        for (int i = 0; i < channelsAmount; ++i) {
            // pool
            const neurologicalValue* __restrict yChannel = cached_y + i * channelSize;

            neurologicalValue* __restrict final_yChannel = final_y + i * finalSize;

            for (int y = 0; y < finalHeight; ++y) {
                const int cY = y*stride;

                const int upClip = std::max(0, -cY);
                const int downClip = std::max(0, (cY+poolHeight)-channelHeight);

                for (int x = 0; x < finalWidth; ++x) {
                    const int cX = x*stride;

                    const int corner = cY*channelWidth + cX;

                    const int leftClip = std::max(0, -cX);
                    const int rightClip = std::max(0, (cX+poolWidth)-channelWidth);

                    neurologicalValue poolValue = 0.0;
                    int pooledSize = 0;

                    for (int pY = upClip; pY < poolHeight-downClip; ++pY) {
                        const int row = corner + pY*channelWidth;
                        for (int pX = leftClip; pX < poolWidth-rightClip; ++pX) {
                            poolValue += yChannel[row + pX];
                            pooledSize += 1;
                        }
                    }

                    // std::cout << poolValue << std::endl;
                    final_yChannel[y*finalWidth + x] = poolValue/pooledSize;
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

    for (int b = 0; b < batchSize; b++)
    {
        const neurologicalValue* __restrict cached_y = cached_output + b * sampleSize;

        neurologicalValue* __restrict final_y = output + b * finalSampleSize;
        int* __restrict mask = picking_mask + b * finalSampleSize;

        for (int i = 0; i < channelsAmount; ++i) {
            // pool
            const neurologicalValue* __restrict yChannel = cached_y + i * channelSize;

            neurologicalValue* __restrict final_yChannel = final_y + i * finalSize;
            int* __restrict mask_channel = mask + i * finalSize;

            for (int y = 0; y < finalHeight; ++y) {
                const int cY = y*stride;

                const int upClip = std::max(0, -cY);
                const int downClip = std::max(0, (cY+poolHeight)-channelHeight);

                for (int x = 0; x < finalWidth; ++x) {
                    const int cX = x*stride;

                    const int corner = cY*channelWidth + cX;

                    const int leftClip = std::max(0, -cX);
                    const int rightClip = std::max(0, (cX+poolWidth)-channelWidth);

                    neurologicalValue poolValue = LARGEST;
                    int targetIndex = 0;

                    for (int pY = upClip; pY < poolHeight-downClip; ++pY) {
                        const int row = corner + pY*channelWidth;
                        for (int pX = leftClip; pX < poolWidth-rightClip; ++pX) {
                            const int yIndex = row + pX;
                            const neurologicalValue& yValue = yChannel[yIndex];
                            if (yValue < poolValue) {
                                poolValue = yValue;
                                targetIndex = yIndex;
                            }
                        }
                    }

                    final_yChannel[y*finalWidth + x] = poolValue;
                    mask_channel[y*finalWidth + x] = targetIndex;
                }
            }
        }
    }
}

// pooling propagation
void picking_pool_propagation(const neurologicalValue* __restrict poolPropagation, neurologicalValue* __restrict propagation
    , const int& batchSize, const int& channelsAmount
    , const int& channelWidth, const int& channelHeight
    , const int& finalWidth, const int& finalHeight
    , const int* __restrict picking_mask)
{
    int channelSize = channelWidth * channelHeight;
    int finalSize = finalWidth * finalHeight;

    int sampleSize = channelsAmount * channelSize;
    int finalSampleSize = channelsAmount * finalSize;

    for (int b = 0; b < batchSize; b++)
    {
        const neurologicalValue* __restrict pool_p = poolPropagation + b * finalSampleSize;
        const int* __restrict mask = picking_mask + b * finalSampleSize;

        neurologicalValue* __restrict p = propagation + b * sampleSize;

        for (int i = 0; i < channelsAmount; ++i) {
            // pool
            const neurologicalValue* __restrict pool_pChannel = pool_p + i * finalSize;
            const int* __restrict mask_channel = mask + i * finalSize;
            
            neurologicalValue* __restrict pChannel = p + i * channelSize;

            for (int y = 0; y < finalHeight; ++y) {
                for (int x = 0; x < finalWidth; ++x) {
                    const int point = y*finalWidth + x;

                    const neurologicalValue& propagationValue = pool_pChannel[point];

                    // std::cout << "propagate: " << y*finalWidth + x << " as " << mask_channel[point] << std::endl;

                    pChannel[mask_channel[point]] = propagationValue;
                }
            }
        }
    }
}
void average_pool_propagation(const neurologicalValue* __restrict poolPropagation, neurologicalValue* __restrict propagation
    , const int& batchSize, const int& channelsAmount
    , const int& channelWidth, const int& channelHeight
    , const int& poolWidth, const int& poolHeight
    , const int& finalWidth, const int& finalHeight
    , const int& stride)
{

    // MAKE SURE DO GET THE BATCH SIZE CORRECTLY BY "finalOutputSize" YOU BAFOOOOM

    // std::cout
    //     << "batchSize: " << batchSize << std::endl
    //     << "channelsAmount: " << channelsAmount << std::endl

    //     << "channelWidth: " << channelWidth << std::endl
    //     << "channelHeight: " << channelHeight << std::endl
        
    //     << "poolWidth: " << poolWidth << std::endl
    //     << "poolHeight: " << poolHeight << std::endl
        
    //     << "finalWidth: " << finalWidth << std::endl
    //     << "finalHeight: " << finalHeight << std::endl

    //     << "stride: " << stride << std::endl;

    int channelSize = channelWidth * channelHeight;
    int finalSize = finalWidth * finalHeight;

    int sampleSize = channelsAmount * channelSize;
    int finalSampleSize = channelsAmount * finalSize;

    for (int b = 0; b < batchSize; b++)
    {
        const neurologicalValue* __restrict pool_p = poolPropagation + b * finalSampleSize;

        neurologicalValue* __restrict p = propagation + b * sampleSize;

        for (int i = 0; i < channelsAmount; ++i) {
            // pool
            const neurologicalValue* __restrict pool_pChannel = pool_p + i * finalSize;

            neurologicalValue* __restrict pChannel = p + i * channelSize;

            for (int y = 0; y < finalHeight; ++y) {
                const int cY = y*stride;

                const int upClip = std::max(0, -cY);
                const int downClip = std::max(0, (cY+poolHeight)-channelHeight);

                const int pooledHeight = poolHeight - (upClip + downClip);

                for (int x = 0; x < finalWidth; ++x) {
                    const int cX = x*stride;

                    const int corner = cY*channelWidth + cX;

                    const int leftClip = std::max(0, -cX);
                    const int rightClip = std::max(0, (cX+poolWidth)-channelWidth);

                    const int pooledSize = pooledHeight * (poolWidth - (leftClip + rightClip));
                    neurologicalValue propagationValue = pool_pChannel[y*finalWidth + x]/pooledSize;

                    for (int pY = upClip; pY < poolHeight-downClip; ++pY) {
                        const int row = corner + pY*channelWidth;
                        for (int pX = leftClip; pX < poolWidth-rightClip; ++pX) {
                            pChannel[row + pX] += propagationValue;
                        }
                    }
                }
            }
        }
    }
}

// back propagation activations
void convolutional_activation_gradient_linear(const neurologicalValue* __restrict propagation, neurologicalValue* __restrict biasesGradient
    , const int& batchSize
    , const int& outputChannels
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
    , const int& outputChannels
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
inline void ensurePacked(size_t needed, neurologicalBuffer& packPropagationBuffer, size_t& packedCap)
{
    if (packedCap >= needed)
        return;

    packPropagationBuffer.resize(needed);

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
    , const int& stride, const int& padding

    , neurologicalBuffer& packedPropagationBuffer
    , size_t& packedCap)
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

        ensurePacked(outputChannels*outputChannelSize, packedPropagationBuffer, packedCap);
        neurologicalValue* __restrict packedPropagation = packedPropagationBuffer.data();
        pack(p, packedPropagation, outputChannelWidth, outputChannelHeight, outputChannelSize, outputChannels, 0);

        // output blocking
        for (int outputStart = 0; outputStart < outputChannels; outputStart+=OUTPUT_BLOCK) {
            const int oMax = std::min(outputStart + OUTPUT_BLOCK, outputChannels);

            const int blockSize = oMax - outputStart;

            // loop through the full propagation by point
            for (int oY = 0; oY < outputChannelHeight; ++oY) {
                const neurologicalValue* __restrict packedPropagationY = packedPropagation + oY*blockSize*outputChannelWidth;

                const int iY = oY*stride - padding;
                
                const int upClip = std::max(0, -iY);
                const int downClip = std::max(0, (iY+kernelHeight)-inputChannelHeight);

                for (int oX = 0; oX < outputChannelWidth; ++oX) {
                    const neurologicalValue* __restrict packedPropagationX = packedPropagationY + oX*blockSize;

                    const int iX = oX*stride - padding;

                    const int xCorner = iY*inputChannelWidth + iX;

                    const int leftClip = std::max(0, -iX);
                    const int rightClip = std::max(0, (iX+kernelWidth)-inputChannelWidth);

                    // accumulate for each propagation in input channels
                    for (int i = 0; i < inputChannels; ++i) {
                        const int weightsIndexing = i*kernelSize*outputChannels;

                        const int inputIndexing = i*inputChannelSize + xCorner;

                        for (int kY = upClip; kY < kernelHeight-downClip; ++kY) {
                            const int wPoint = kY*kernelWidth*outputChannels
                                + leftClip*outputChannels
                                + outputStart;

                            const neurologicalValue* w_row = weights
                                + weightsIndexing

                                + wPoint;

                            neurologicalValue* wG_row = weightsGradient
                                + weightsIndexing

                                + wPoint;

                            const int xIndexPoint = kY*inputChannelWidth + leftClip;

                            neurologicalValue* __restrict inputPropagationRow = inputP + inputIndexing + xIndexPoint;
                            const neurologicalValue* __restrict cached_xRow = cached_x + inputIndexing + xIndexPoint;

                            for (int kX = leftClip; kX < kernelWidth-rightClip; ++kX) {
                                const neurologicalValue& cached_xValue = *cached_xRow;
                                
                                neurologicalValue g = 0.0;

                                for (int o = 0; o < blockSize; ++o) {
                                    const neurologicalValue& wValue = w_row[o];

                                    g += wValue*packedPropagationX[o+outputStart];

                                    wG_row[o] += (double)(cached_xValue*packedPropagationX[o+outputStart]);
                                }

                                *inputPropagationRow += g;

                                inputPropagationRow += 1;
                                cached_xRow += 1;
                                
                                w_row += outputChannels;
                                wG_row += outputChannels;
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
    const neurologicalValue decayMultiplier = 1 - 0.001*rate;

    for (int i = 0; i < gradientSize; ++i)
    {
        if (weightsGradient[i] > 25) {
            std::cout << "weight gradient value: " << weightsGradient[i] << std::endl;

            throw std::runtime_error("LARGE WEIGHTS GRADIENT FOR CONVOLUTION!");
        }
        weights[i] = weights[i]*decayMultiplier - weightsGradient[i] * rate;
    }
}
void update_biases_CONVO(neurologicalValue* __restrict biases, const neurologicalValue* __restrict biasesGradient
    , const int& gradientSize
    , const neurologicalValue& rate)
{
    const neurologicalValue decayMultiplier = 1 - 0.001*rate;

    for (int i = 0; i < gradientSize; ++i)
        biases[i] = biases[i]*decayMultiplier - biasesGradient[i] * rate;
}

void ConvolutionalLayer::ForwardPass(ConvolutionalLayer* convolutionalLayer, const neurologicalConstantSpan input, const neurologicalSpan output, const int& batchSize) {    
    // std::cout << input.size() << std::endl;
    
    neurologicalSpan targetOutputSpan = output;
    if (convolutionalLayer->channelsPooler->poolerType != ChannelsPoolerTypes::NONE) {
        // std::cout << "prepare for pooling" << std::endl;
        if (convolutionalLayer->channelsPooler->cached_output.size() != convolutionalLayer->outputSize*batchSize) {
            convolutionalLayer->channelsPooler->cached_output.resize(convolutionalLayer->outputSize*batchSize);
        }
        // std::cout << "prepared for pooling i believe" << std::endl;
        zeroOutList(convolutionalLayer->channelsPooler->cached_output);
        // std::cout << "start pooling?? " << targetOutputSpan.size() << ", " << convolutionalLayer->finalOutputSize*batchSize << std::endl;
        targetOutputSpan = convolutionalLayer->channelsPooler->cached_output;
        // std::cout << "okay?? " << targetOutputSpan.size() << ", " << convolutionalLayer->outputSize*batchSize << std::endl;
    }

    convolutionalLayer->cached_input = input;

    convolutional_weights_compute(
        input.data(),
        targetOutputSpan.data(),

        convolutionalLayer->weights.data(),
        
        batchSize,

        convolutionalLayer->inputChannels,
        convolutionalLayer->outputChannels,

        convolutionalLayer->kernelWidth,
        convolutionalLayer->kernelHeight,

        convolutionalLayer->inputChannelWidth,
        convolutionalLayer->inputChannelHeight,
        
        convolutionalLayer->outputChannelWidth,
        convolutionalLayer->outputChannelHeight,
        
        convolutionalLayer->stride,
        convolutionalLayer->padding
    );
    switch(convolutionalLayer->activationFunction->activationType) {
        case ActivationFunctionTypes::RELU: {
            ReLU* relu = static_cast<ReLU*>(convolutionalLayer->activationFunction.get());
            if (relu->cached_output.size() != targetOutputSpan.size()) {
                relu->cached_output.resize(targetOutputSpan.size());
            }
            convolutional_activation_relu(
                targetOutputSpan.data(),

                convolutionalLayer->biases.data(),

                batchSize,
                    
                convolutionalLayer->outputChannels,

                convolutionalLayer->outputChannelWidth,
                convolutionalLayer->outputChannelHeight,
            
                relu->cached_output.data(),
                relu->leakage
            );
            break;
        }
        default: {
            convolutional_activation_linear(
                targetOutputSpan.data(),

                convolutionalLayer->biases.data(),

                batchSize,
                    
                convolutionalLayer->outputChannels,

                convolutionalLayer->outputChannelWidth,
                convolutionalLayer->outputChannelHeight
            );
            break;
        }
    }

    // try pooling
    switch (convolutionalLayer->channelsPooler->poolerType) {
        case ChannelsPoolerTypes::MAX: {
            Max* maxPool = static_cast<Max*>(convolutionalLayer->channelsPooler.get());
            // std::cout << "prepare max pool" << std::endl;
            if (maxPool->picking_mask.size() != output.size()) {
                maxPool->picking_mask.resize(output.size());
            }
            // std::cout << "max pool" << std::endl;
            max_pool(
                targetOutputSpan.data(),
                output.data(),

                batchSize,
                convolutionalLayer->outputChannels,

                convolutionalLayer->outputChannelWidth,
                convolutionalLayer->outputChannelHeight,
                
                maxPool->poolingWidth,
                maxPool->poolingHeight,
                
                convolutionalLayer->finalChannelWidth,
                convolutionalLayer->finalChannelHeight,

                maxPool->stride,

                maxPool->picking_mask.data()
            );
            break;
        }
        case ChannelsPoolerTypes::AVERAGE: {
            Average* averagePool = static_cast<Average*>(convolutionalLayer->channelsPooler.get());

            average_pool(
                targetOutputSpan.data(),
                output.data(),

                batchSize,
                convolutionalLayer->outputChannels,

                convolutionalLayer->outputChannelWidth,
                convolutionalLayer->outputChannelHeight,
                
                averagePool->poolingWidth,
                averagePool->poolingHeight,
                
                convolutionalLayer->finalChannelWidth,
                convolutionalLayer->finalChannelHeight,

                averagePool->stride
            );
            break;
        }
        case ChannelsPoolerTypes::MIN: {
            Min* minPool = static_cast<Min*>(convolutionalLayer->channelsPooler.get());
            if (minPool->picking_mask.size() != output.size()) {
                minPool->picking_mask.resize(output.size());
            }

            min_pool(
                targetOutputSpan.data(),
                output.data(),

                batchSize,
                convolutionalLayer->outputChannels,

                convolutionalLayer->outputChannelWidth,
                convolutionalLayer->outputChannelHeight,
                
                minPool->poolingWidth,
                minPool->poolingHeight,
                
                convolutionalLayer->finalChannelWidth,
                convolutionalLayer->finalChannelHeight,

                minPool->stride,

                minPool->picking_mask.data()
            );
            break;
        }
    }
}

void ConvolutionalLayer::BackPropagation(ConvolutionalLayer* convolutionalLayer, const neurologicalSpan propagation, const neurologicalSpan inputPropagation, const neurologicalValue& learnRate, const int& batchSize) {
    neurologicalValue rate = learnRate/batchSize;

    neurologicalSpan targetPropagationSpan = propagation;

    if (convolutionalLayer->channelsPooler->poolerType != ChannelsPoolerTypes::NONE) {
        if (convolutionalLayer->channelsPooler->processedPropagation.size() != convolutionalLayer->outputSize*batchSize) {
            convolutionalLayer->channelsPooler->processedPropagation.resize(convolutionalLayer->outputSize*batchSize);
        }
        zeroOutList(convolutionalLayer->channelsPooler->processedPropagation);

        targetPropagationSpan = convolutionalLayer->channelsPooler->processedPropagation;
    }

    // try pooling
    switch (convolutionalLayer->channelsPooler->poolerType) {
        case ChannelsPoolerTypes::MAX: {
            Max* maxPool = static_cast<Max*>(convolutionalLayer->channelsPooler.get());

            picking_pool_propagation(
                propagation.data(),
                targetPropagationSpan.data(),

                batchSize,
                convolutionalLayer->outputChannels,

                convolutionalLayer->outputChannelWidth,
                convolutionalLayer->outputChannelHeight,
                
                convolutionalLayer->finalChannelWidth,
                convolutionalLayer->finalChannelHeight,

                maxPool->picking_mask.data()
            );
            break;
        }
        case ChannelsPoolerTypes::AVERAGE: {
            Average* averagePool = static_cast<Average*>(convolutionalLayer->channelsPooler.get());

            average_pool_propagation(
                propagation.data(),
                targetPropagationSpan.data(),

                batchSize,
                convolutionalLayer->outputChannels,

                convolutionalLayer->outputChannelWidth,
                convolutionalLayer->outputChannelHeight,
                
                averagePool->poolingWidth,
                averagePool->poolingHeight,
                
                convolutionalLayer->finalChannelWidth,
                convolutionalLayer->finalChannelHeight,

                averagePool->stride
            );
            break;
        }
        case ChannelsPoolerTypes::MIN: {
            Min* minPool = static_cast<Min*>(convolutionalLayer->channelsPooler.get());

            picking_pool_propagation(
                propagation.data(),
                targetPropagationSpan.data(),

                batchSize,
                convolutionalLayer->outputChannels,

                convolutionalLayer->outputChannelWidth,
                convolutionalLayer->outputChannelHeight,
                
                convolutionalLayer->finalChannelWidth,
                convolutionalLayer->finalChannelHeight,

                minPool->picking_mask.data()
            );
            break;
        }
    }

    switch(convolutionalLayer->activationFunction->activationType) {
        case ActivationFunctionTypes::RELU: {
            ReLU* relu = static_cast<ReLU*>(convolutionalLayer->activationFunction.get());
            convolutional_activation_gradient_relu(
                targetPropagationSpan.data(),
                convolutionalLayer->biasesGradient.data(),

                batchSize,
                    
                convolutionalLayer->outputChannels,

                convolutionalLayer->outputChannelWidth,
                convolutionalLayer->outputChannelHeight,
            
                relu->cached_output.data(),
                relu->leakage
            );
            break;
        }
        default: {
            convolutional_activation_gradient_linear(
                targetPropagationSpan.data(),
                convolutionalLayer->biasesGradient.data(),

                batchSize,
                    
                convolutionalLayer->outputChannels,

                convolutionalLayer->outputChannelWidth,
                convolutionalLayer->outputChannelHeight
            );
            break;
        }
    }
    convolutional_propagation(
        targetPropagationSpan.data(),

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
        convolutionalLayer->outputChannelHeight,
        
        convolutionalLayer->stride,
        convolutionalLayer->padding,

        convolutionalLayer->packedPropagation,
        convolutionalLayer->packedCap
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