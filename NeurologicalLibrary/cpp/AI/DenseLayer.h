// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "NeurologicalComponent.h"

#include <vector>
#include <optional>
#include <unordered_map>

using namespace std;

/**
 * 
 */

struct DenseNeuron
{
public:
	vector<NeurologicalParameter> weights;
	NeurologicalParameter bias;
};

typedef vector<DenseNeuron> DenseNeurons;

class DenseLayer : public INeurologicalComponent
{
public:
	DenseNeurons neurons;

	unordered_map<string, const neurologicalListValues*> passedInputs;

	int inputSize;
	int neuronsSize;

	DenseLayer();
	DenseLayer(int inputSize, int neuronsSize);

	void initialize(NeurologicalComponentInitializationInfo info) override;

	int getOutputSize() override;

	void forwardPass(const neurologicalListValues& _input, neurologicalListValues& _output, const string& sampleIndex);
	void backPropagation(const neurologicalListValues& _propagation, neurologicalListValues& _inputPropagation, const string& sampleIndex);
	void applyDerivatives(derivativesApplyingInfo applyingInfo) override;
};
