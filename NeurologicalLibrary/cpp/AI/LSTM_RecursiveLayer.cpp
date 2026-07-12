// Fill out your copyright notice in the Description page of Project Settings.


#include "LSTM_RecursiveLayer.h"
#include "NeurologicalComponent.h"

void LSTM_RecursiveLayer::makeGates()
{
    this->forgetGate = make_shared<DenseLayer>(this->concatenatedSize, this->outputSize);
    // this->forgetGateActivation = make_shared<Sigmoid>();
    this->inputGate = make_shared<DenseLayer>(this->concatenatedSize, this->outputSize);
    // this->inputGateActivation = make_shared<Sigmoid>();
    this->candidateMemoryGate = make_shared<DenseLayer>(this->concatenatedSize, this->outputSize);
    // this->candidateMemoryGateActivation = make_shared<Tanh>();
    this->outputGate = make_shared<DenseLayer>(this->concatenatedSize, this->outputSize);
    // this->outputGateActivation = make_shared<Sigmoid>();

    // this->hiddenStateActivation = make_shared<Tanh>();
}

LSTM_RecursiveLayer::LSTM_RecursiveLayer()
{
    this->inputSize = 0;
    this->outputSize = 0;
    this->concatenatedSize = 0;

    this->makeGates();
}
LSTM_RecursiveLayer::LSTM_RecursiveLayer(int inputSize, int outputSize)
{
    this->inputSize = inputSize;
    this->outputSize = outputSize;
    this->concatenatedSize = this->inputSize+this->outputSize;

    this->makeGates();
}

void LSTM_RecursiveLayer::initialize(NeurologicalComponentInitializationInfo info)
{
	this->optimizer = info.optimizer;
    
    this->forgetGate->initialize(info);
    // this->forgetGateActivation->initialize(info);
    this->inputGate->initialize(info);
    // this->inputGateActivation->initialize(info);
    this->candidateMemoryGate->initialize(info);
    // this->candidateMemoryGateActivation->initialize(info);
    this->outputGate->initialize(info);
    // this->outputGateActivation->initialize(info);
    
    // this->hiddenStateActivation->initialize(info);
    
    this->plainHiddenState = neurologicalListValues(this->outputSize);
    this->plainCache = { ._hiddenState = &plainHiddenState
        , ._memoryCell = neurologicalListValues(this->outputSize) };
}

int LSTM_RecursiveLayer::getOutputSize() {
    return this->outputSize;
}

pair<string, int> LSTM_RecursiveLayer::turnSampleIndexToSeriesPosition(const string& sampleIndex)
{
    vector<string> splitedIndex = splitStringByString(sampleIndex, "-s");

    string seriesIndex = splitedIndex[0];
    int stepIndex = stoi(splitedIndex[1]);

    return pair<string, int>(seriesIndex, stepIndex);
}
const LSTM_Cache& LSTM_RecursiveLayer::getStepCache(const string& seriesIndex, const int& stepIndex)
{
    if (this->seriesCaches.contains(seriesIndex)) {
        if (checkInBounds(this->seriesCaches[seriesIndex], stepIndex)) {
            return this->seriesCaches[seriesIndex][stepIndex];
        }
    }
    return this->plainCache;
}
void LSTM_RecursiveLayer::addStepCache(const string& seriesIndex, const LSTM_Cache& cache)
{
    if (!this->seriesCaches.contains(seriesIndex)) {
        this->seriesCaches[seriesIndex] = vector<LSTM_Cache>();
    }
    this->seriesCaches[seriesIndex].push_back(cache);
}

void LSTM_RecursiveLayer::forwardPass(const neurologicalListValues& _input, neurologicalListValues& _output, const string& sampleIndex)
{
    pair<string, int> position = this->turnSampleIndexToSeriesPosition(sampleIndex);
    const LSTM_Cache& previousStepCache = this->getStepCache(position.first, position.second-1);

    this->passedConcatenatedInputs[sampleIndex] = joinLists<neurologicalValue>({*previousStepCache._hiddenState, _input});
    const neurologicalListValues& concatenatedInput = this->passedConcatenatedInputs[sampleIndex];

    LSTM_Cache newCache = {};
    newCache._hiddenState = &_output;
    newCache._memoryCell = neurologicalListValues(this->outputSize);

    newCache._forgetGate_activation = neurologicalListValues(this->outputSize);
    newCache._inputGate_activation = neurologicalListValues(this->outputSize);
    newCache._candidateMemoryGate_activation = neurologicalListValues(this->outputSize);
    newCache._outputGate_activation = neurologicalListValues(this->outputSize);
    newCache._memoryCell_activation = neurologicalListValues(this->outputSize);

    this->forgetGate->forwardPass(concatenatedInput, newCache._forgetGate_activation, sampleIndex);
    Sigmoid::__activation(newCache._forgetGate_activation, newCache._forgetGate_activation);
    
    this->inputGate->forwardPass(concatenatedInput, newCache._inputGate_activation, sampleIndex);
    Sigmoid::__activation(newCache._inputGate_activation, newCache._inputGate_activation);

    this->candidateMemoryGate->forwardPass(concatenatedInput, newCache._candidateMemoryGate_activation, sampleIndex);
    Tanh::__activation(newCache._candidateMemoryGate_activation, newCache._candidateMemoryGate_activation);

    this->outputGate->forwardPass(concatenatedInput, newCache._outputGate_activation, sampleIndex);
    Sigmoid::__activation(newCache._outputGate_activation, newCache._outputGate_activation);
;
    for (int i = 0; i < this->outputSize; i++) {
        newCache._memoryCell[i] = (previousStepCache._memoryCell[i] * newCache._forgetGate_activation[i])
        + (newCache._inputGate_activation[i]*newCache._candidateMemoryGate_activation[i]);
    }

    // neurologicalListValues tanhedMemoryCell(newCache._memoryCell.size());
    Tanh::__activation(newCache._memoryCell, newCache._memoryCell_activation);
    for (int i = 0; i < this->outputSize; i++) {
        // final output
        _output[i] = newCache._outputGate_activation[i]*newCache._memoryCell_activation[i];
    }

    this->addStepCache(position.first, newCache);
}

// MAKE SURE BACKPROPAGATION IS DONE THROUGH A REVERSED TRAJECTORY (like policy gradient and what not)
void LSTM_RecursiveLayer::backPropagation(const neurologicalListValues& _propagation, neurologicalListValues& _inputPropagation, const string& sampleIndex)
{
    pair<string, int> position = this->turnSampleIndexToSeriesPosition(sampleIndex);
    const LSTM_Cache& stepCache = this->getStepCache(position.first, position.second);
    const LSTM_Cache& previousStepCache = this->getStepCache(position.first, position.second-1);

    int size = _propagation.size();

    neurologicalListValues memoryCellPassingGradient(size);
    if (this->passingMemoryCellGradients.contains(position.first)) {
        memoryCellPassingGradient = this->passingMemoryCellGradients[position.first];
    }
    neurologicalListValues propagationAddition(size);
    if (this->passingPropagationAdditions.contains(position.first)) {
        propagationAddition = this->passingPropagationAdditions[position.first];
    }
    
    neurologicalListValues newMemoryCellPassingGradient(size);

    neurologicalListValues outputGatePropagation(size);

    neurologicalListValues memeoryCellPropagation(size);
    
    neurologicalListValues forgetGatePropagation(size);
    neurologicalListValues inputGatePropagation(size);
    neurologicalListValues candidateMemoryGatePropagation(size);

    for (int i = 0; i < _propagation.size(); i++) {
        neurologicalValue propagationValue = _propagation[i]+propagationAddition[i];
        neurologicalValue memoryCellActivation = stepCache._memoryCell_activation[i];
 
        outputGatePropagation[i] = (propagationValue * memoryCellActivation)
        * Sigmoid::__derivative(stepCache._outputGate_activation[i]);

        memeoryCellPropagation[i] = ((propagationValue * stepCache._outputGate_activation[i])
        * Tanh::__derivative(memoryCellActivation)) + memoryCellPassingGradient[i];

        forgetGatePropagation[i] =
        (memeoryCellPropagation[i] * previousStepCache._memoryCell[i])
        * Sigmoid::__derivative(stepCache._forgetGate_activation[i]);

        inputGatePropagation[i] =
        (memeoryCellPropagation[i] * stepCache._candidateMemoryGate_activation[i])
        * Sigmoid::__derivative(stepCache._inputGate_activation[i]);

        candidateMemoryGatePropagation[i] =
        (memeoryCellPropagation[i] * stepCache._inputGate_activation[i])
        * Tanh::__derivative(stepCache._candidateMemoryGate_activation[i]);

        newMemoryCellPassingGradient[i] = memeoryCellPropagation[i] * stepCache._forgetGate_activation[i];
    }
    this->passingMemoryCellGradients[position.first] = newMemoryCellPassingGradient;
    neurologicalListValues forgetGateGradient(this->concatenatedSize);
    this->forgetGate->backPropagation(forgetGatePropagation, forgetGateGradient, sampleIndex);

    neurologicalListValues inputGateGradient(this->concatenatedSize);
    (this->inputGate->backPropagation(inputGatePropagation, inputGateGradient, sampleIndex));

    neurologicalListValues candidateMemoryGateGradient(this->concatenatedSize);
    (this->candidateMemoryGate->backPropagation(candidateMemoryGatePropagation, candidateMemoryGateGradient, sampleIndex));

    neurologicalListValues outputGateGradient(this->concatenatedSize);
    (this->outputGate->backPropagation(outputGatePropagation, outputGateGradient, sampleIndex));

    neurologicalListValues previousStepPropagationAddition(this->outputSize);

    // split and sum gradients
    for (int i = 0; i < this->concatenatedSize; i++) {
        if (i >= this->outputSize) {
            _inputPropagation[i-this->outputSize] =
            forgetGateGradient[i] +
            inputGateGradient[i] +
            candidateMemoryGateGradient[i] +
            outputGateGradient[i];
        } else {
            previousStepPropagationAddition[i] =
            forgetGateGradient[i] +
            inputGateGradient[i] +
            candidateMemoryGateGradient[i] +
            outputGateGradient[i];
        }
    }

    this->passingPropagationAdditions[position.first] = previousStepPropagationAddition;
}

void LSTM_RecursiveLayer::applyDerivatives(derivativesApplyingInfo applyingInfo)
{
    this->forgetGate->applyDerivatives(applyingInfo);
    this->inputGate->applyDerivatives(applyingInfo);
    this->candidateMemoryGate->applyDerivatives(applyingInfo);
    this->outputGate->applyDerivatives(applyingInfo);

    this->passedConcatenatedInputs.clear();
    this->seriesCaches.clear();
    this->passingMemoryCellGradients.clear();
    this->passingPropagationAdditions.clear();
}