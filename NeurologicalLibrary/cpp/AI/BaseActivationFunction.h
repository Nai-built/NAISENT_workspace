// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

//#include "CoreMinimal.h"
#include "NeurologicalComponent.h"

#include <vector>
#include <optional>
#include <unordered_map>

using namespace std;

/**
 *
 */

class BaseActivationFunction : public INeurologicalComponent
{
public:
	unordered_map<string, const neurologicalListValues*> passedInputs__list;
	unordered_map<string, const neurologicalListValues*> passedOutputs__list;
	
	unordered_map<string, const neurologicalTensorChannels*> passedInputs__tensor;
	unordered_map<string, const neurologicalTensorChannels*> passedOutputs__tensor;

	void initialize(NeurologicalComponentInitializationInfo info) override;

	virtual void activation(const neurologicalListValues& _listInput, neurologicalListValues& _listOutput) = 0;
	virtual void gradient(const neurologicalListValues& _listPropagation, const neurologicalListValues& _listInput, const neurologicalListValues& _listOutput, neurologicalListValues& _listInputPropagatoin) = 0;

	int getOutputSize() override;

	void forwardPass__list(const neurologicalListValues& _input, neurologicalListValues& _output, const string& sampleIndex);
	void backPropagation__list(const neurologicalListValues& _propagation, neurologicalListValues& _inputPropagation, const string& sampleIndex);

	void forwardPass__tensor(const neurologicalTensorChannels& _input, neurologicalTensorChannels& _output, const string& sampleIndex);
	void backPropagation__tensor(const neurologicalTensorChannels& _propagation, neurologicalTensorChannels& _inputPropagation, const string& sampleIndex);

	void applyDerivatives(derivativesApplyingInfo applyingInfo) override;
};
