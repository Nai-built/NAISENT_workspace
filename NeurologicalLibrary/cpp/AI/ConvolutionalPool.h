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

class ConvolutionalPool : public INeurologicalComponent
{
private:
    int getProcessedInputsAmount(const neurologicalMatrixValues& inputChannel, const int& _inputY, const int& _inputX);

    neurologicalValue poolPoint(const neurologicalMatrixValues& inputChannel, const int& _y, const int& _x);
    void poolChannel(const neurologicalMatrixValues& inputChannel, neurologicalMatrixValues& outputChannel, const int& matrixHeight, const int& matrixWidth);
    
    void propagatePoint(const neurologicalValue& propagationValue, const neurologicalMatrixValues& inputChannel, neurologicalMatrixValues& inputPropagationChannel
        , const int& _y, const int& _x);
    void propagateChannel(const neurologicalMatrixValues& propagationChannel, const neurologicalMatrixValues& inputChannel, neurologicalMatrixValues& inputPropagationChannel);
public:
	unordered_map<string, const neurologicalTensorChannels*> passedInputs;

    string poolingType;
    int poolWidth;
    int poolHeight;
    int stride;

    int totalSize;

    ConvolutionalPool(string _poolingType, int _poolWidth, int _poolHeight, int _stride);

	void initialize(NeurologicalComponentInitializationInfo info) override;

	int getOutputSize() override;

	void forwardPass(const neurologicalTensorChannels& _input, neurologicalTensorChannels& _output, const string& sampleIndex);
	void backPropagation(const neurologicalTensorChannels& _propagation, neurologicalTensorChannels& _inputPropagation, const string& sampleIndex);
	void applyDerivatives(derivativesApplyingInfo applyingInfo) override;
};