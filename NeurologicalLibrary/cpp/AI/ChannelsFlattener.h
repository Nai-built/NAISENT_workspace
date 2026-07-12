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
 
class ChannelsFlattener : public INeurologicalComponent
{
public:
	unordered_map<string, const neurologicalTensorChannels*> passedInputs;

    ChannelsFlattener();

	void initialize(NeurologicalComponentInitializationInfo info) override;

	int getOutputSize() override;

	void forwardPass(const neurologicalTensorChannels& _input, neurologicalListValues& _output, const string& sampleIndex);
	void backPropagation(const neurologicalListValues& _propagation, neurologicalTensorChannels& _inputPropagation, const string& sampleIndex);
	void applyDerivatives(derivativesApplyingInfo applyingInfo) override;
};