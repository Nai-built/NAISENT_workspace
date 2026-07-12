// Fill out your copyright notice in the Description page of Project Settings.


#include "DenseLayer.h"
#include "NeurologicalComponent.h"

#include <iostream>
#include <memory>
#include <random>

#include <stdio.h>
#include <stdlib.h>

using namespace std;

//static random_device rd;
//static default_random_engine r_engine(rd());

DenseLayer::DenseLayer(int _inputSize, int _neuronsSize)
{
	inputSize = _inputSize;
	neuronsSize = _neuronsSize;
}

void DenseLayer::initialize(NeurologicalComponentInitializationInfo info)
{
	vector<DenseNeuron> _neurons(neuronsSize);

	neurologicalValue weights_initialization_boundary = (neurologicalValue)sqrt(6 / ((double)(inputSize) + (double)(neuronsSize)));
	
	//uniform_real_distribution<neurologicalValue> distribution(-weights_initialization_boundary, weights_initialization_boundary);

	this->optimizer = info.optimizer;
	this->clippingMethod = info.clippingMethod;

	for (int i = 0; i < neuronsSize; i++) {
		vector<NeurologicalParameter> weights(inputSize);
		for (int j = 0; j < inputSize; j++) {
			neurologicalValue _value = getRandom(-weights_initialization_boundary, weights_initialization_boundary);
			if (info.optimizer.has_value()) {
				weights[j] = NeurologicalParameter(_value, this->optimizer.value()->initializeParameter());
			}
			else {
				weights[j] = NeurologicalParameter(_value);
			}
		}
		neurologicalValue _value = 0.0;
		NeurologicalParameter bias;
		if (info.optimizer.has_value()) {
			bias = NeurologicalParameter(_value, this->optimizer.value()->initializeParameter());
		}
		else {
			bias = NeurologicalParameter(_value);
		}
		_neurons[i] = DenseNeuron {
			.weights = weights,
			.bias = bias,
		};
	}

	neurons = _neurons;
}

int DenseLayer::getOutputSize()
{
	return this->neurons.size();
}

void DenseLayer::forwardPass(const neurologicalListValues& _input, neurologicalListValues& _output, const string& sampleIndex)
{
	passedInputs[sampleIndex] = &_input;

	for (int i = 0; i < neuronsSize; i++) {
		const DenseNeuron& neuron = neurons[i];
		neurologicalValue outputValue = 0.0;
		for (int j = 0; j < inputSize; j++) {
			outputValue += (_input[j] * neuron.weights[j].value);
		}
		outputValue += neuron.bias.value;

		_output[i] = outputValue;
	}
}
void DenseLayer::backPropagation(const neurologicalListValues& _propagation, neurologicalListValues& _inputPropagation, const string& sampleIndex)
{
	const neurologicalListValues& input = *passedInputs[sampleIndex];

	for (int i = 0; i < neuronsSize; i++) {
		neurologicalValue neuronDerivative = _propagation[i];

		DenseNeuron& neuron = neurons[i];
		for (int j = 0; j < inputSize; j++) {
			neurologicalValue inputPropagationValue = neuronDerivative * neuron.weights[j].value;

			_inputPropagation[j] += inputPropagationValue;

			neurologicalValue weightDerivative = neuronDerivative * input[j];
			neuron.weights[j].addCalculatedDerivative(weightDerivative);
		}

		neurologicalValue biasDerivative = neuronDerivative;
		neuron.bias.addCalculatedDerivative(biasDerivative);
	}
}

void DenseLayer::applyDerivatives(derivativesApplyingInfo applyingInfo)
{
	if (this->clippingMethod.has_value()) {
		auto NORM_clippingMethod = dynamic_pointer_cast<NORM_ClippingMethod>(this->clippingMethod.value());
		if (NORM_clippingMethod) {
			NORM_clippingMethod->poweredDerivativesSum = 0;
			for (auto& neuron : this->neurons) {
				NORM_clippingMethod->poweredDerivativesSum += pow(neuron.bias.checkCalculatedDerivative(), 2);
				for (auto& weight : neuron.weights) {
					NORM_clippingMethod->poweredDerivativesSum += pow(weight.checkCalculatedDerivative(), 2);
				}
			}
		}
	}

	for (int i = 0; i < neuronsSize; i++) {
		for (int j = 0; j < inputSize; j++) {
			neurons[i].weights[j].applyCalculatedDerivatives(applyingInfo, this->optimizer, this->clippingMethod);
		}
		neurons[i].bias.applyCalculatedDerivatives(applyingInfo, this->optimizer, this->clippingMethod);
	}

	this->passedInputs.clear();
}

//DenseLayer::~DenseLayer()
//{
//	/*for (DenseNeuron* neuron : this->neurons)
//	{
//		delete neuron;
//	}*/
//}