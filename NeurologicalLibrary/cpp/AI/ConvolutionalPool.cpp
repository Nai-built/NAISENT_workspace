// Fill out your copyright notice in the Description page of Project Settings.


#include "ConvolutionalPool.h"
#include "NeurologicalComponent.h"

#include <iostream>
#include <memory>
#include <random>
#include <algorithm>

#include <stdio.h>
#include <stdlib.h>

using namespace std;

int ConvolutionalPool::getProcessedInputsAmount(const neurologicalMatrixValues& inputChannel, const int& _inputY, const int& _inputX) {
    int processedInputsAmount = totalSize;

    processedInputsAmount -= (max(0, (int)(((_inputY+1)+poolHeight-1)-inputChannel.size())))*poolWidth;
    processedInputsAmount -= (max(0, (int)(((_inputX+1)+poolWidth-1)-inputChannel[0].size())))*poolHeight;

    return processedInputsAmount;
}

neurologicalValue ConvolutionalPool::poolPoint(const neurologicalMatrixValues& inputChannel, const int& _y, const int& _x) {
    const int _inputY = _y*stride;
    const int _inputX = _x*stride;
    
    neurologicalValue outputValue = 0.0;

    if (poolingType == "AVERAGE") {
        int processedInputsAmount = this->getProcessedInputsAmount(inputChannel, _inputY, _inputX);

        // properly get input [ STARTED ]
        for (int _yP = 0; _yP < poolHeight; _yP++) {
            int _finalY = _inputY+_yP;
		    bool inYBounds = checkInBounds(inputChannel, _inputY);
            if (!inYBounds) {
                continue;
            }
            for (int _xP = 0; _xP < poolWidth; _xP++) {
                int _finalX = _inputX+_xP;
                neurologicalValue inputValue;
                if (inYBounds && checkInBounds(inputChannel[_inputY], _inputX)) {
                    inputValue = inputChannel[_inputY][_inputX];
                } else {
                    continue;
                }
        // properly get input [ END ]

                outputValue += inputValue;
            }
        }
        outputValue /= processedInputsAmount;
    } else if (poolingType == "MAX" || poolingType == "MIN") {
        bool startedPooling = false;
        
        // properly get input [ STARTED ]
        for (int _yP = 0; _yP < poolHeight; _yP++) {
            int _finalY = _inputY+_yP;
		    bool inYBounds = checkInBounds(inputChannel, _inputY);
            if (!inYBounds) {
                continue;
            }
            for (int _xP = 0; _xP < poolWidth; _xP++) {
                int _finalX = _inputX+_xP;
                neurologicalValue inputValue;
                if (inYBounds && checkInBounds(inputChannel[_inputY], _inputX)) {
                    inputValue = inputChannel[_inputY][_inputX];
                } else {
                    continue;
                }
        // properly get input [ END ]

                bool checked = false;
                if (startedPooling) {
                    if (poolingType == "MAX") {
                        if (inputValue > outputValue) {
                            checked = true;
                        }
                    } else if (poolingType == "MIN") {
                        if (inputValue < outputValue) {
                            checked = true;
                        }
                    }
                } else {
                    checked = true;
                }

                if (checked) {
                    outputValue = inputValue;
                }

                startedPooling = true;
            }
        }
    }

    return outputValue;
}

void ConvolutionalPool::poolChannel(const neurologicalMatrixValues& inputChannel, neurologicalMatrixValues& outputChannel, const int& matrixHeight, const int& matrixWidth) {
    for (int _y = 0; _y < matrixHeight; _y++) {
        outputChannel[_y] = neurologicalListValues(matrixWidth);

        for (int _x = 0; _x < matrixWidth; _x++) {
            outputChannel[_y][_x] = this->poolPoint(inputChannel, _y, _x);
        }
    }
}

void ConvolutionalPool::propagatePoint(const neurologicalValue& propagationValue, const neurologicalMatrixValues& inputChannel, neurologicalMatrixValues& inputPropagationChannel
        , const int& _y, const int& _x) {
    const int _inputY = _y*stride;
    const int _inputX = _x*stride;
    
    if (poolingType == "AVERAGE") {
        int processedInputsAmount = this->getProcessedInputsAmount(inputChannel, _inputY, _inputX);

        // properly check input [ STARTED ]
        for (int _yP = 0; _yP < poolHeight; _yP++) {
            int _finalY = _inputY+_yP;
		    bool inYBounds = checkInBounds(inputChannel, _inputY);
            if (!inYBounds) {
                continue;
            }
            for (int _xP = 0; _xP < poolWidth; _xP++) {
                int _finalX = _inputX+_xP;
                if (!checkInBounds(inputChannel[_inputY], _inputX)) {
                    continue;
                }
        // properly check input [ END ]

                inputPropagationChannel[_finalY][_finalX] += propagationValue/processedInputsAmount;
            }
        }
    } else if (poolingType == "MAX" || poolingType == "MIN") {
        neurologicalValue targetValue = 0.0;
        bool startedPooling = false;
        int targetY = 0;
        int targetX = 0;

        // properly get input [ STARTED ]
        for (int _yP = 0; _yP < poolHeight; _yP++) {
            int _finalY = _inputY+_yP;
		    bool inYBounds = checkInBounds(inputChannel, _inputY);
            if (!inYBounds) {
                continue;
            }
            for (int _xP = 0; _xP < poolWidth; _xP++) {
                int _finalX = _inputX+_xP;
                neurologicalValue inputValue;
                if (inYBounds && checkInBounds(inputChannel[_inputY], _inputX)) {
                    inputValue = inputChannel[_inputY][_inputX];
                } else {
                    continue;
                }
        // properly get input [ END ]

                bool checked = false;
                if (startedPooling) {
                    if (poolingType == "MAX") {
                        if (inputValue > targetValue) {
                            checked = true;
                        }
                    } else if (poolingType == "MIN") {
                        if (inputValue < targetValue) {
                            checked = true;
                        }
                    }
                } else {
                    checked = true;
                }

                if (checked) {
                    targetValue = inputValue;
                    targetY = _finalY;
                    targetX = _finalX;
                }

                startedPooling = true;
            }
        }

        inputPropagationChannel[targetY][targetX] += propagationValue;
    }
}
void ConvolutionalPool::propagateChannel(const neurologicalMatrixValues& propagationChannel, const neurologicalMatrixValues& inputChannel, neurologicalMatrixValues& inputPropagationChannel) {
    for (int _y = 0; _y < propagationChannel.size(); _y++) {
        const neurologicalListValues& propagationRow = propagationChannel[_y];
        for (int _x = 0; _x < propagationRow.size(); _x++) {
            this->propagatePoint(propagationRow[_x], inputChannel, inputPropagationChannel, _y, _x);
        }
    }
}

ConvolutionalPool::ConvolutionalPool(string _poolingType, int _poolWidth, int _poolHeight, int _stride)
{
    this->poolingType = _poolingType;
    this->poolWidth = _poolWidth;
    this->poolHeight = _poolHeight;
    this->stride = _stride;

    this->totalSize = poolHeight*poolWidth;
}

void ConvolutionalPool::initialize(NeurologicalComponentInitializationInfo info)
{
    // empty
}

int ConvolutionalPool::getOutputSize()
{
	return -1; // INPUT DEPENDANT
}

void ConvolutionalPool::forwardPass(const neurologicalTensorChannels& _input, neurologicalTensorChannels& _output, const string& sampleIndex)
{
    passedInputs[sampleIndex] = &_input;

    int matrixHeight = ceil((_input[0].size()-poolHeight)/stride) + 1;
    int matrixWidth = ceil((_input[0][0].size()-poolWidth)/stride) + 1;

    for (int i = 0; i < _input.size(); i++) {
        _output[i] = neurologicalMatrixValues(matrixHeight);
        this->poolChannel(_input[i], _output[i], matrixHeight, matrixWidth);
    }
}

void ConvolutionalPool::backPropagation(const neurologicalTensorChannels& _propagation, neurologicalTensorChannels& _inputPropagation, const string& sampleIndex)
{
	const neurologicalTensorChannels& input = *passedInputs[sampleIndex];

	for (int i = 0; i < input.size(); i++) {
		_inputPropagation[i] = neurologicalMatrixValues(input[i].size());
		
		for (int j = 0; j < input[i].size(); j++) {
			_inputPropagation[i][j] = neurologicalListValues(input[i][j].size());
		}
	}

    for (int i = 0; i < input.size(); i++) {
        this->propagateChannel(_propagation[i], input[i], _inputPropagation[i]);
    }
}

void ConvolutionalPool::applyDerivatives(derivativesApplyingInfo applyingInfo)
{
	this->passedInputs.clear();
}