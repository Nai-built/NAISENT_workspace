#include <iostream>
#include <assert.h>

#include "ComponentsChain.h"
#include "DenseLayer.h"
#include "ConvolutionalLayer.h"
#include "LSTM_RecursiveLayer.h"

void initializeChainBuffer(std::vector<neurologicalBuffer>& chainBuffers, const int& i, const neurologicalConstantSpan input, INeurologicalComponent* _component, const int& batchSize
    , const int& totalSamples)
{
    switch(_component->componentType) {
        case NeurologicalComponentTypes::DENSE_LAYER: {
            DenseLayer* denseLayer = static_cast<DenseLayer*>(_component);

            assert(input.size() % denseLayer->inputSize == 0 && "BATCH SIZE DOESN'T MATCH DENSE LAYER EXPECTATION");

            int batchOutputSize = batchSize*denseLayer->outputSize;

            if (chainBuffers[i].size() != batchOutputSize) {
                chainBuffers[i].resize(batchOutputSize);
            }
            break;
        }
        case NeurologicalComponentTypes::CONVOLUTIONAL_LAYER: {
            ConvolutionalLayer* convolutionalLayer = static_cast<ConvolutionalLayer*>(_component);

            assert(input.size() % convolutionalLayer->inputSize == 0 && "BATCH SIZE DOESN'T MATCH CONVOLUTIONAL LAYER EXPECTATION");

            int batchOutputSize = batchSize*convolutionalLayer->finalOutputSize;

            if (chainBuffers[i].size() != batchOutputSize) {
                chainBuffers[i].resize(batchOutputSize);
            }
            break;
        }
        case NeurologicalComponentTypes::RECURSIVE_LSTM: {
            LSTM_RecursiveLayer* lstm_recursiveLayer = static_cast<LSTM_RecursiveLayer*>(_component);

            assert(input.size() % lstm_recursiveLayer->inputSize == 0 && "BATCH SIZE DOESN'T MATCH LSTM LAYER EXPECTATION");

            int batchOutputSize = 0;
            if (lstm_recursiveLayer->castBeyond) {
                batchOutputSize = totalSamples*lstm_recursiveLayer->outputSize;
            } else {
                batchOutputSize = batchSize*lstm_recursiveLayer->outputSize;
            }

            if (chainBuffers[i].size() != batchOutputSize) {
                chainBuffers[i].resize(batchOutputSize);
            }
            break;
        }
        case NeurologicalComponentTypes::COMPONENTS_CHAIN: {
            break;
        }
    }
}
void initializeChainPropagation(std::vector<neurologicalBuffer>& chainPropagations, const int& i, const neurologicalConstantSpan propagation, INeurologicalComponent* _component, const int& batchSize
    , const int& totalSamples)
{
    switch(_component->componentType) {
        case NeurologicalComponentTypes::DENSE_LAYER: {
            DenseLayer* denseLayer = static_cast<DenseLayer*>(_component);

            assert(propagation.size() % denseLayer->outputSize == 0 && "BATCH SIZE DOESN'T MATCH DENSE LAYER PROPAGATION");

            int batchInputSize = batchSize*denseLayer->inputSize;

            if (chainPropagations[i].size() != batchInputSize) {
                chainPropagations[i].resize(batchInputSize);
            }
            break;
        }
        case NeurologicalComponentTypes::CONVOLUTIONAL_LAYER: {
            ConvolutionalLayer* convolutionalLayer = static_cast<ConvolutionalLayer*>(_component);

            assert(propagation.size() % convolutionalLayer->finalOutputSize == 0 && "BATCH SIZE DOESN'T MATCH CONVOLUTIONAL LAYER PROPAGATION");

            int batchInputSize = batchSize*convolutionalLayer->inputSize;

            if (chainPropagations[i].size() != batchInputSize) {
                chainPropagations[i].resize(batchInputSize);
            }
            break;
        }
        case NeurologicalComponentTypes::RECURSIVE_LSTM: {
            LSTM_RecursiveLayer* lstm_recursiveLayer = static_cast<LSTM_RecursiveLayer*>(_component);

            assert(propagation.size() % lstm_recursiveLayer->outputSize == 0 && "BATCH SIZE DOESN'T MATCH LSTM LAYER PROPAGATION");

            int batchInputSize = lstm_recursiveLayer->cached_input.size();

            if (chainPropagations[i].size() != batchInputSize) {
                chainPropagations[i].resize(batchInputSize);
            }
            break;
        }
        case NeurologicalComponentTypes::COMPONENTS_CHAIN: {
            break;
        }
    }
}

void activateChainElement(INeurologicalComponent* _component, const neurologicalConstantSpan input, const neurologicalSpan output, const int& batchSize
    , const lengths seriesLengths, const int& totalSamples)
{
    switch(_component->componentType) {
        case NeurologicalComponentTypes::DENSE_LAYER: {
            DenseLayer* denseLayer = static_cast<DenseLayer*>(_component);
            DenseLayer::ForwardPass(denseLayer, input, output, batchSize);
            break;
        }
        case NeurologicalComponentTypes::CONVOLUTIONAL_LAYER: {
            ConvolutionalLayer* convolutionalLayer = static_cast<ConvolutionalLayer*>(_component);
            ConvolutionalLayer::ForwardPass(convolutionalLayer, input, output, batchSize);
            break;
        }
        case NeurologicalComponentTypes::RECURSIVE_LSTM: {
            LSTM_RecursiveLayer* lstm_recursiveLayer = static_cast<LSTM_RecursiveLayer*>(_component);
            LSTM_RecursiveLayer::ForwardPass(lstm_recursiveLayer, input, output, batchSize, seriesLengths, totalSamples);
            break;
        }
        case NeurologicalComponentTypes::COMPONENTS_CHAIN: {
            std::cout << "cannot forward pass for a chain inside a chain" << std::endl;
            break;
        }
    }
}

void adjustChainElement(INeurologicalComponent* _component, const neurologicalSpan propagation, const neurologicalSpan inputPropagation, const neurologicalValue& learnRate, const int& batchSize
    , const lengths seriesLengths, const int& totalSamples)
{
    switch(_component->componentType) {
        case NeurologicalComponentTypes::DENSE_LAYER: {
            DenseLayer* denseLayer = static_cast<DenseLayer*>(_component);
            DenseLayer::BackPropagation(denseLayer, propagation, inputPropagation, learnRate, batchSize);
            break;
        }
        case NeurologicalComponentTypes::CONVOLUTIONAL_LAYER: {
            ConvolutionalLayer* convolutionalLayer = static_cast<ConvolutionalLayer*>(_component);
            ConvolutionalLayer::BackPropagation(convolutionalLayer, propagation, inputPropagation, learnRate, batchSize);
            break;
        }
        case NeurologicalComponentTypes::RECURSIVE_LSTM: {
            LSTM_RecursiveLayer* lstm_recursiveLayer = static_cast<LSTM_RecursiveLayer*>(_component);
            LSTM_RecursiveLayer::BackPropagation(lstm_recursiveLayer, propagation, inputPropagation, learnRate, batchSize, seriesLengths, totalSamples);
            break;
        }
        case NeurologicalComponentTypes::COMPONENTS_CHAIN: {
            std::cout << "cannot back propagate for a chain inside a chain" << std::endl;
            break;
        }
    }
}

ComponentsChain::ComponentsChain() {
    this->componentType = NeurologicalComponentTypes::COMPONENTS_CHAIN;

    this->componentsChain = std::vector<NeurologicalComponent_UNIQUE>();
}

void ComponentsChain::Initialize(ComponentsChain* chain) {
    // EMPTY
}

void ComponentsChain::ActivateChain(ComponentsChain* chain, const neurologicalConstantSpan input, const neurologicalSpan output, const int& batchSize
    , const lengths seriesLengths, const int& totalSamples)
{
    if (chain->chainBuffers.size() != chain->componentsChain.size()-1) {
        chain->chainBuffers.resize(chain->componentsChain.size()-1);
    }

    for (int i = 0; i < chain->componentsChain.size(); ++i) {
        if (i == 0) {
            initializeChainBuffer(chain->chainBuffers, i, input, chain->componentsChain[i].get(), batchSize, totalSamples);
            zeroOutList(chain->chainBuffers[i]);
            activateChainElement(chain->componentsChain[i].get(), input, chain->chainBuffers[i], batchSize, seriesLengths, totalSamples);
        } else if (i >= chain->componentsChain.size()-1) {
            activateChainElement(chain->componentsChain[i].get(), chain->chainBuffers[i-1], output, batchSize, seriesLengths, totalSamples);
        } else {
            initializeChainBuffer(chain->chainBuffers, i, chain->chainBuffers[i-1], chain->componentsChain[i].get(), batchSize, totalSamples);
            zeroOutList(chain->chainBuffers[i]);
            activateChainElement(chain->componentsChain[i].get(), chain->chainBuffers[i-1], chain->chainBuffers[i], batchSize, seriesLengths, totalSamples);
        }
    }
}

void ComponentsChain::AdjustChain(ComponentsChain* chain, const neurologicalSpan propagation, const neurologicalSpan inputPropagation, const neurologicalValue& learnRate, const int& batchSize
    , const lengths seriesLengths, const int& totalSamples)
{    
    if (chain->chainPropagations.size() != chain->componentsChain.size()-1) {
        chain->chainPropagations.resize(chain->componentsChain.size()-1);
    }
    
    for (int i = chain->componentsChain.size()-1; i >= 0 ; --i) {
        int pI = i-1;
        if (i >= chain->componentsChain.size()-1) {
            initializeChainPropagation(chain->chainPropagations, pI /*input propagation index*/, propagation /*propagation*/, chain->componentsChain[i].get(), batchSize, totalSamples);
            zeroOutList(chain->chainPropagations[pI]);
            adjustChainElement(chain->componentsChain[i].get(), propagation, chain->chainPropagations[pI], learnRate, batchSize, seriesLengths, totalSamples);
        } else if (i == 0) {
            adjustChainElement(chain->componentsChain[i].get(), chain->chainPropagations[i], inputPropagation, learnRate, batchSize, seriesLengths, totalSamples);
        } else {
            initializeChainPropagation(chain->chainPropagations, pI /*input propagation index*/, chain->chainPropagations[i] /*propagation*/, chain->componentsChain[i].get(), batchSize, totalSamples);
            zeroOutList(chain->chainPropagations[pI]);
            adjustChainElement(chain->componentsChain[i].get(), chain->chainPropagations[i], chain->chainPropagations[pI], learnRate, batchSize, seriesLengths, totalSamples);
        }
    }
}