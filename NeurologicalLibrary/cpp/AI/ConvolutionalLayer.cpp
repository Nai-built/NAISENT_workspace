// Fill out your copyright notice in the Description page of Project Settings.


#include "ConvolutionalLayer.h"
#include "NeurologicalComponent.h"

#include <iostream>
#include <memory>
#include <random>

#include <stdio.h>
#include <stdlib.h>

using namespace std;

// ALL cordinates are just corners btw

neurologicalValue ConvolutionalLayer::captureKernelConvolution(const int& _xCord, const int& _yCord, const Kernel& kernel, const neurologicalMatrixValues& input) {
	neurologicalValue kernelOutput = 0.0;
	
	for (int _y = 0; _y < kernel.size(); _y++) {
		const vector<NeurologicalParameter>& kernelRow = kernel[_y];
		
		int _inputY = _y+_yCord;
		bool inYBounds = checkInBounds(input, _inputY);

		for (int _x = 0; _x < kernelRow.size(); _x++) {
			int _inputX = _x+_xCord;
			if (inYBounds && checkInBounds(input[_inputY], _inputX)) {
				kernelOutput += kernelRow[_x].value * input[_inputY][_inputX];
			}
		}
	}

	return kernelOutput;
}

void ConvolutionalLayer::slideKernelThroughInput(const Kernel& kernel, const neurologicalMatrixValues& input, neurologicalMatrixValues& filterMatrix) {		
	for (int _y = 0; _y < filterMatrix.size(); _y++) {
		if (checkInBounds(input, _y)) {
			for (int _x = 0; _x < filterMatrix[_y].size(); _x++) {
				filterMatrix[_y][_x] += this->captureKernelConvolution((_x*stride)-padding, (_y*stride)-padding, kernel, input);
			}
		}
	}
}

void ConvolutionalLayer::propagateKernel(const int& _resultX, const int& _resultY, Kernel& kernel, const neurologicalValue& propagationValue, const neurologicalMatrixValues& input, neurologicalMatrixValues& inputPropagation)
{
	// getting input corner index through result index
	const int _xCord = (_resultX*stride)-padding;
	const int _yCord = (_resultY*stride)-padding;

	for (int _y = 0; _y < kernel.size(); _y++) {
		vector<NeurologicalParameter>& kernelRow = kernel[_y];

		int _inputY = _y+_yCord;
		bool inYBounds = checkInBounds(input, _inputY);

		for (int _x = 0; _x < kernelRow.size(); _x++) {
			neurologicalValue inputValue = 0.0;

			int _inputX = _x+_xCord;
			if (inYBounds && checkInBounds(input[_inputY], _inputX)) {
				inputPropagation[_inputY][_inputX] += propagationValue * kernelRow[_x].value;
				inputValue = input[_inputY][_inputX];
			}
			
			neurologicalValue _derivative = propagationValue * inputValue;
			kernelRow[_x].addCalculatedDerivative(_derivative);
		}
	}
}

ConvolutionalLayer::ConvolutionalLayer(int _inputChannelsSize, int _filterChannelsSize
	, int _padding, int _stride
	, int _kernelWidth, int _kernelHeight)
{
	this->inputChannelsSize = _inputChannelsSize;
	this->filterChannelsSize = _filterChannelsSize;

	this->padding = _padding;
	this->stride = _stride;

	this->kernelWidth = _kernelWidth;
	this->kernelHeight = _kernelHeight;
}

void ConvolutionalLayer::initialize(NeurologicalComponentInitializationInfo info)
{
	vector<ConvolutionalFilter> _filters(this->filterChannelsSize);

	int inputSize = kernelWidth*kernelHeight*inputChannelsSize;
	neurologicalValue weights_initialization_boundary = ((neurologicalValue)sqrt(6 / ((double)(inputSize) + (double)(filterChannelsSize)))) * 2.0f;

	this->optimizer = info.optimizer;
	this->clippingMethod = info.clippingMethod;

	for (int i = 0; i < filterChannelsSize; i++) {
		vector<Kernel> kernels(inputChannelsSize);
		for (int j = 0; j < inputChannelsSize; j++) {
			Kernel _kernel(kernelHeight);
			for (int _y = 0; _y < kernelHeight; _y++) {
				vector<NeurologicalParameter> _kernelRow(kernelWidth);
				for (int _x = 0; _x < kernelWidth; _x++) {
					neurologicalValue _value = getRandom(-weights_initialization_boundary, weights_initialization_boundary);
					if (info.optimizer.has_value()) {
						_kernelRow[_x] = NeurologicalParameter(_value, this->optimizer.value()->initializeParameter());
					}
					else {
						_kernelRow[_x] = NeurologicalParameter(_value);
					}
				}
				_kernel[_y] = _kernelRow;
			}
			kernels[j] = _kernel;
		}
		neurologicalValue _value = 0.0;
		NeurologicalParameter bias;
		if (info.optimizer.has_value()) {
			bias = NeurologicalParameter(_value, this->optimizer.value()->initializeParameter());
		}
		else {
			bias = NeurologicalParameter(_value);
		}
		_filters[i] = ConvolutionalFilter {
			.kernels = kernels,
			.bias = bias,
		};
	}

	filters = _filters;
}

int ConvolutionalLayer::getOutputSize()
{
	return -1; // INPUT DEPENDANT
}

void ConvolutionalLayer::forwardPass(const neurologicalTensorChannels& _input, neurologicalTensorChannels& _output, const string& sampleIndex)
{
	passedInputs[sampleIndex] = &_input;

	int matrixHeight = ceil((_input[0].size()-(kernelHeight)+(padding*2))/stride) + 1;
	int matrixWidth = ceil((_input[0][0].size()-(kernelWidth)+(padding*2))/stride) + 1;
	
	// cout << "passing" << endl;

	for (int i = 0; i < filterChannelsSize; i++) {
		// cout << "filtering" << endl;

		_output[i] = neurologicalMatrixValues(matrixHeight);

		const ConvolutionalFilter& filter = filters[i];

		for (int _y = 0; _y < matrixHeight; _y++) {
			_output[i][_y] = neurologicalListValues(matrixWidth);
			// just initializing the matrix with the filter bias value
			for (int _x = 0; _x < matrixWidth; _x++) {
				_output[i][_y][_x] = filter.bias.value;
			}
		}

		for (int j = 0; j < inputChannelsSize; j++) {
			this->slideKernelThroughInput(filter.kernels[j], _input[j], _output[i]);
		}
	}
}

void ConvolutionalLayer::backPropagation(const neurologicalTensorChannels& _propagation, neurologicalTensorChannels& _inputPropagation, const string& sampleIndex)
{
	const neurologicalTensorChannels& input = *passedInputs[sampleIndex];
	
	for (int i = 0; i < input.size(); i++) {
		_inputPropagation[i] = neurologicalMatrixValues(input[i].size());
		
		for (int j = 0; j < input[i].size(); j++) {
			_inputPropagation[i][j] = neurologicalListValues(input[i][j].size());
		}
	}

	for (int i = 0; i < filterChannelsSize; i++) {
		const neurologicalMatrixValues& filterChannelPropagation = _propagation[i];

		ConvolutionalFilter& filter = filters[i];

		neurologicalValue biasDerivative = 0.0;

		for (int _y = 0; _y < filterChannelPropagation.size(); _y++) {
			const neurologicalListValues& propagationRow = filterChannelPropagation[_y];
			for (int _x = 0; _x < propagationRow.size(); _x++) {
				biasDerivative += propagationRow[_x];

				for (int j = 0; j < filter.kernels.size(); j++) {
					const neurologicalMatrixValues& kernelChannelInput = input[j];
					neurologicalMatrixValues& kernelChannelInputPropagation = _inputPropagation[j];
					this->propagateKernel(_x, _y, filter.kernels[j], propagationRow[_x], kernelChannelInput, kernelChannelInputPropagation);
				}
			}
		}

		filter.bias.addCalculatedDerivative(biasDerivative);
	}
}

void ConvolutionalLayer::applyDerivatives(derivativesApplyingInfo applyingInfo)
{
	if (this->clippingMethod.has_value()) {
		auto NORM_clippingMethod = dynamic_pointer_cast<NORM_ClippingMethod>(this->clippingMethod.value());
		if (NORM_clippingMethod) {
			NORM_clippingMethod->poweredDerivativesSum = 0;
			for (auto& filter : this->filters) {
				NORM_clippingMethod->poweredDerivativesSum += pow(filter.bias.checkCalculatedDerivative(), 2);
				for (auto& kernel : filter.kernels) {
					for (auto& kernelRow : kernel) {
						for (auto& kernelWeight : kernelRow) {
							NORM_clippingMethod->poweredDerivativesSum += pow(kernelWeight.checkCalculatedDerivative(), 2);
						}
					}
				}
			}
		}
	}

	for (auto& filter : this->filters) {
		filter.bias.applyCalculatedDerivatives(applyingInfo, this->optimizer, this->clippingMethod);
		for (auto& kernel : filter.kernels) {
			for (auto& kernelRow : kernel) {
				for (auto& kernelWeight : kernelRow) {
					kernelWeight.applyCalculatedDerivatives(applyingInfo, this->optimizer, this->clippingMethod);
				}
			}
		}
	}

	this->passedInputs.clear();
}