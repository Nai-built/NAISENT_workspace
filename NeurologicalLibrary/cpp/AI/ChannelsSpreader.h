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
 
 // DOES NOT SUPPORT BACK PROPAGATION
class ChannelsSpreader : public INeurologicalComponent
{
public:
    int expectedChannelsAmount;
    int expectedChannelsWidth;
    int expectedChannelsHeight;

	ChannelsSpreader();
    ChannelsSpreader(int _a, int _w, int _h);

	void initialize(NeurologicalComponentInitializationInfo info) override;

	int getOutputSize() override;

	void forwardPass(const neurologicalListValues& _input, neurologicalTensorChannels& _output, const string& sampleIndex);
	void backPropagation(const neurologicalListValues& _propagation, neurologicalTensorChannels& _inputPropagation, const string& sampleIndex);
	void applyDerivatives(derivativesApplyingInfo applyingInfo) override;
};