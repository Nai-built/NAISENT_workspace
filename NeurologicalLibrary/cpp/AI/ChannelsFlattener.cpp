// Fill out your copyright notice in the Description page of Project Settings.


#include "ChannelsFlattener.h"
#include "NeurologicalComponent.h"

#include <iostream>
#include <memory>
#include <random>

#include <stdio.h>
#include <stdlib.h>

using namespace std;

ChannelsFlattener::ChannelsFlattener()
{

}

void ChannelsFlattener::initialize(NeurologicalComponentInitializationInfo info)
{
    // empty
}

int ChannelsFlattener::getOutputSize()
{
	return -1; // INPUT DEPENDANT
}

void ChannelsFlattener::forwardPass(const neurologicalTensorChannels& _input, neurologicalListValues& _output, const string& sampleIndex)
{
    passedInputs[sampleIndex] = &_input;
    // squesh into one list
    for (int i = 0; i < _input.size(); i++) {
        const neurologicalMatrixValues& inputcChannel = _input[i];
        for (int _y = 0; _y < inputcChannel.size(); _y++) {
            const neurologicalListValues& inputRow = inputcChannel[_y];
            for (int _x = 0; _x < inputRow.size(); _x++) {
                const neurologicalValue& inputValue = inputRow[_x];
                _output.push_back(inputValue);
            }
        }
    }

    // cout << "flattened: " << _output.size() << endl;
}

void ChannelsFlattener::backPropagation(const neurologicalListValues& _propagation, neurologicalTensorChannels& _inputPropagation, const string& sampleIndex)
{
	const neurologicalTensorChannels& input = *passedInputs[sampleIndex];

    // spread propagation into channels identical to the input channels
    int propagationIndex = 0;
    for (int i = 0; i < input.size(); i++) {
        const neurologicalMatrixValues& inputcChannel = input[i];
        _inputPropagation.push_back(neurologicalMatrixValues(inputcChannel.size()));
        for (int _y = 0; _y < inputcChannel.size(); _y++) {
            const neurologicalListValues& inputRow = inputcChannel[_y];
            _inputPropagation[i][_y] = neurologicalListValues(inputRow.size());
            for (int _x = 0; _x < inputRow.size(); _x++) {
                _inputPropagation[i][_y][_x] = _propagation[propagationIndex];
                propagationIndex+=1;
            }
        }
    }
}

void ChannelsFlattener::applyDerivatives(derivativesApplyingInfo applyingInfo)
{
	this->passedInputs.clear();
}