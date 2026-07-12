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

 // matrix of Neurological Parameters
typedef vector<vector<NeurologicalParameter>> Kernel;

struct ConvolutionalFilter {
    vector<Kernel> kernels;
    NeurologicalParameter bias;
};

typedef vector<ConvolutionalFilter> ConvolutionalFilters;

class ConvolutionalLayer : public INeurologicalComponent
{
private:
	neurologicalValue captureKernelConvolution(const int& _xCord, const int& _yCord, const Kernel& kernel, const neurologicalMatrixValues& input);
	void slideKernelThroughInput(const Kernel& kernel, const neurologicalMatrixValues& input, neurologicalMatrixValues& filterMatrix);

	void propagateKernel(const int& _resultX, const int& _resultY, Kernel& kernel, const neurologicalValue& propagationValue, const neurologicalMatrixValues& input, neurologicalMatrixValues& inputPropagation);
public:
	ConvolutionalFilters filters;

	unordered_map<string, const neurologicalTensorChannels*> passedInputs;

	int inputChannelsSize;
	int filterChannelsSize;

	int padding;
	int stride;

	int kernelWidth;
	int kernelHeight;

	ConvolutionalLayer();
	ConvolutionalLayer(int _inputChannelsSize, int _filterChannelsSize
		, int _padding, int _stride
		, int _kernelWdidth, int _kernelHeight);

	void initialize(NeurologicalComponentInitializationInfo info) override;

	int getOutputSize() override;

	void forwardPass(const neurologicalTensorChannels& _input, neurologicalTensorChannels& _output, const string& sampleIndex);
	void backPropagation(const neurologicalTensorChannels& _propagation, neurologicalTensorChannels& _inputPropagation, const string& sampleIndex);
	void applyDerivatives(derivativesApplyingInfo applyingInfo) override;
};
