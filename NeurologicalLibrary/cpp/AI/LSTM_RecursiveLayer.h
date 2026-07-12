// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "NeurologicalComponent.h"
#include "DenseLayer.h"
#include "ActivationFunctions/Sigmoid.h"
#include "ActivationFunctions/Tanh.h"

#include <vector>
#include <optional>
#include <unordered_map>
#include <iostream>

struct LSTM_Cache {
	neurologicalListValues* _hiddenState; // the actual output
	neurologicalListValues _memoryCell;
	
	neurologicalListValues _forgetGate_activation;
	neurologicalListValues _inputGate_activation;
	neurologicalListValues _candidateMemoryGate_activation;
	neurologicalListValues _outputGate_activation;
	neurologicalListValues _memoryCell_activation;
};

class LSTM_RecursiveLayer : public INeurologicalComponent
{
private:
	neurologicalListValues plainHiddenState;
	LSTM_Cache plainCache;
public:
    shared_ptr<DenseLayer> forgetGate;
    shared_ptr<DenseLayer> inputGate;
    shared_ptr<DenseLayer> candidateMemoryGate;
    shared_ptr<DenseLayer> outputGate;
	
	// indexed by sampleIndex
	unordered_map<string, neurologicalListValues> passedConcatenatedInputs;

	// indexed by seriesIndex
	unordered_map<string, vector<LSTM_Cache>> seriesCaches;
	unordered_map<string, neurologicalListValues> passingMemoryCellGradients;
	unordered_map<string, neurologicalListValues> passingPropagationAdditions;

	int inputSize;
	int outputSize;
    int concatenatedSize;

	void makeGates();

	LSTM_RecursiveLayer();
	LSTM_RecursiveLayer(int inputSize, int neuronsSize);

	void initialize(NeurologicalComponentInitializationInfo info) override;

	int getOutputSize() override;

	pair<string, int> turnSampleIndexToSeriesPosition(const string& sampleIndex);
	const LSTM_Cache& getStepCache(const string& seriesIndex, const int& stepIndex);
	void addStepCache(const string& seriesIndex, const LSTM_Cache& cache);

	void forwardPass(const neurologicalListValues& _input, neurologicalListValues& _output, const string& sampleIndex);
	void backPropagation(const neurologicalListValues& _propagation, neurologicalListValues& _inputPropagation, const string& sampleIndex);
	void applyDerivatives(derivativesApplyingInfo applyingInfo) override;
};