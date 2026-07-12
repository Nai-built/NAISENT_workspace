// Fill out your copyright notice in the Description page of Project Settings.


#include "BaseActivationFunction.h"

#include <iostream>
#include <memory>
#include <random>
#include <optional>

#include <stdio.h>
#include <stdlib.h>

using namespace std;

void BaseActivationFunction::initialize(NeurologicalComponentInitializationInfo info)
{
	// EMPTY
}

int BaseActivationFunction::getOutputSize()
{
	return -1;
}
// template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
// template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

void BaseActivationFunction::forwardPass__list(const neurologicalListValues& _input, neurologicalListValues& _output, const string& sampleIndex)
{
	this->passedInputs__list[sampleIndex] = &_input;
	this->activation(_input, _output);
	this->passedOutputs__list[sampleIndex] = &_output;
}
void BaseActivationFunction::backPropagation__list(const neurologicalListValues& _propagation, neurologicalListValues& _inputPropagation, const string& sampleIndex)
{
	const neurologicalListValues& input = *this->passedInputs__list[sampleIndex];
	const neurologicalListValues& output = *this->passedOutputs__list[sampleIndex];
	this->gradient(_propagation, input, output, _inputPropagation);
}

void BaseActivationFunction::forwardPass__tensor(const neurologicalTensorChannels& _input, neurologicalTensorChannels& _output, const string& sampleIndex)
{
	this->passedInputs__tensor[sampleIndex] = &_input;
	for (int i = 0; i < _input.size(); i++) {
		_output[i] = neurologicalMatrixValues(_input[i].size());
		for (int _y = 0; _y < _input[i].size(); _y++) {
			_output[i][_y] = neurologicalListValues(_input[i][_y].size());
			this->activation(_input[i][_y], _output[i][_y]);
		}
	}
	this->passedOutputs__tensor[sampleIndex] = &_output;
}
void BaseActivationFunction::backPropagation__tensor(const neurologicalTensorChannels& _propagation, neurologicalTensorChannels& _inputPropagation, const string& sampleIndex)
{
	const neurologicalTensorChannels& input = *this->passedInputs__tensor[sampleIndex];
	const neurologicalTensorChannels& output = *this->passedOutputs__tensor[sampleIndex];
	for (int i = 0; i < _propagation.size(); i++) {
		_inputPropagation[i] = neurologicalMatrixValues(_propagation[i].size());
		for (int _y = 0; _y < _propagation[i].size(); _y++) {
			_inputPropagation[i][_y] = neurologicalListValues(_propagation[i][_y].size());
			this->gradient(_propagation[i][_y], input[i][_y], output[i][_y], _inputPropagation[i][_y]);
		}
	}
}

// NeurologicalPassingValues BaseActivationFunction::forwardPass(NeurologicalPassingValues _input, const string& sampleIndex)
// {
// 	{
// 		lock_guard<mutex> lock(this->input_mutex);

// 		this->passedInputs[sampleIndex] = _input;
// 	}

// 	NeurologicalPassingValues output;

// 	if (holds_alternative<neurologicalListValues>(_input)) {
// 		const neurologicalListValues& _activation = this->activation(get<neurologicalListValues>(_input));
// 		output = _activation;
// 	}
// 	else if (holds_alternative<neurologicalTensorChannels>(_input)) {
// 		const neurologicalTensorChannels& channelsInput = get<neurologicalTensorChannels>(_input);

// 		neurologicalTensorChannels channelsOutput(channelsInput.size());

// 		for (int i = 0; i < channelsInput.size(); i++) {
// 			channelsOutput[i] = neurologicalMatrixValues(channelsInput[i].size());
				
// 			for (int _y = 0; _y < channelsInput[i].size(); _y++) {
// 				channelsOutput[i][_y] = this->activation(channelsInput[i][_y] /*a Row*/);
// 			}
// 		}

// 		output = channelsOutput;
// 	}
// 	else if (holds_alternative<neurologicalGraphNodes>(_input)) {
// 		neurologicalGraphNodes graphOutput;

// 		const neurologicalGraphNodes& nodesInput = get<neurologicalGraphNodes>(_input);

// 		for (pair<string, neurologicalGraphNode> element : nodesInput) {
// 			neurologicalListValues _activation = this->activation(element.second.values);

// 			graphOutput[element.first] = { .values = _activation, .bridges = element.second.bridges };
// 		}

// 		output = graphOutput;
// 	}

// 	// TEMP
// 	{
// 		lock_guard<mutex> lock(this->output_mutex);

// 		this->passedOutputs[sampleIndex] = output;
// 	}
// 	return output;
// }

// NeurologicalPassingValues BaseActivationFunction::backPropagation(NeurologicalPassingValues _propagation, const string& sampleIndex)
// {
// 	// TEMP
// 	NeurologicalPassingValues input;
// 	NeurologicalPassingValues output;

// 	{
// 		lock_guard<mutex> lock(this->input_mutex);

// 		input = (this->passedInputs[sampleIndex]);
// 	}
// 	{
// 		lock_guard<mutex> lock(this->output_mutex);

// 		output = (this->passedOutputs[sampleIndex]);
// 	}

// 	NeurologicalPassingValues gradient;

// 	if (holds_alternative<neurologicalListValues>(_propagation)) {
// 		//cout << "LIST\n";
// 		const neurologicalListValues& _gradient = this->gradient(get<neurologicalListValues>(_propagation)
// 			, get<neurologicalListValues>(input)
// 			, get<neurologicalListValues>(output));
// 		gradient = _gradient;
// 	}
// 	else if (holds_alternative<neurologicalTensorChannels>(_propagation)) {
// 		const neurologicalTensorChannels& channelsPropagation = get<neurologicalTensorChannels>(_propagation);
// 		const neurologicalTensorChannels& channelsInput = get<neurologicalTensorChannels>(input);
// 		const neurologicalTensorChannels& channelsOutput = get<neurologicalTensorChannels>(output);

// 		neurologicalTensorChannels channelsGradient(channelsPropagation.size());

// 		for (int i = 0; i < channelsPropagation.size(); i++) {
// 			channelsGradient[i] = neurologicalMatrixValues(channelsPropagation[i].size());
				
// 			for (int _y = 0; _y < channelsPropagation[i].size(); _y++) {
// 				channelsGradient[i][_y] = this->gradient(channelsPropagation[i][_y]
// 					, channelsInput[i][_y]
// 					, channelsOutput[i][_y]);
// 			}
// 		}

// 		gradient = channelsGradient;
// 	}
// 	else if (holds_alternative<neurologicalGraphNodes>(_propagation)) {
// 		//cout << "GRAPH\n";

// 		neurologicalGraphNodes graphGradient;

// 		neurologicalGraphNodes& nodesPropagation = get<neurologicalGraphNodes>(_propagation);
// 		neurologicalGraphNodes& nodesInput = get<neurologicalGraphNodes>(input);
// 		neurologicalGraphNodes& nodesOutput = get<neurologicalGraphNodes>(output);

// 		for (pair<string, neurologicalGraphNode> element : nodesPropagation) {
// 			neurologicalListValues _gradient = this->gradient(nodesPropagation[element.first].values
// 				, nodesInput[element.first].values
// 				, nodesOutput[element.first].values);

// 			graphGradient[element.first] = { .values = _gradient, .bridges = element.second.bridges };
// 		}

// 		gradient = graphGradient;
// 	}

// 	return gradient;
// }

void BaseActivationFunction::applyDerivatives(derivativesApplyingInfo applyingInfo)
{
	this->passedInputs__list.clear();
	this->passedOutputs__list.clear();
	this->passedInputs__tensor.clear();
	this->passedOutputs__tensor.clear();
}