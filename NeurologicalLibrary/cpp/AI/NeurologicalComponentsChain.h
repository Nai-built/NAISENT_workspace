// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "NeurologicalComponent.h"

#include <variant>
#include <vector>
#include <optional>
#include <memory>
#include <functional>

using namespace std;

using NeurologicalPassingValues = variant<neurologicalListValues, neurologicalMatrixValues, neurologicalTensorChannels, neurologicalGraphNodes>;

/**
 * 
 */

typedef function<void(string sampleIndex)> componentCallback;

class NeurologicalComponentsChain : public INeurologicalComponent
{
private:
	unordered_map<string, unordered_map<int, neurologicalListValues>> listsBuffer;
	unordered_map<string, unordered_map<int, neurologicalTensorChannels>> tensorsBuffer;
	unordered_map<string, unordered_map<int, neurologicalListValues>> listsPropagation;
	unordered_map<string, unordered_map<int, neurologicalTensorChannels>> tensorsPropagation;

	vector<componentCallback> forwardPassProgram;
	vector<componentCallback> backPropagationProgram;

	void buildComponentForwardPass(vector<componentCallback>& program, const int& _index, INeurologicalComponentPTR component);
	void buildComponentBackPropagation(vector<componentCallback>& program, const int& _index, INeurologicalComponentPTR component);
public:
	vector<INeurologicalComponentPTR> chain;

	NeurologicalComponentsChain (vector<INeurologicalComponentPTR> _chain);

	void initialize(NeurologicalComponentInitializationInfo info) override;

	int getOutputSize() override;

	NeurologicalPassingValues forwardPass(NeurologicalPassingValues _input, const string& sampleIndex);
	NeurologicalPassingValues backPropagation(NeurologicalPassingValues _propagation, const string& sampleIndex);
	void applyDerivatives(derivativesApplyingInfo applyingInfo) override;
};
