// Fill out your copyright notice in the Description page of Project Settings.


#include "ChannelsSpreader.h"
#include "NeurologicalComponent.h"

#include <iostream>
#include <memory>
#include <random>

#include <stdio.h>
#include <stdlib.h>

using namespace std;

ChannelsSpreader::ChannelsSpreader()
{
    this->expectedChannelsAmount = 0;
    this->expectedChannelsWidth = 0;
    this->expectedChannelsHeight = 0;
}
ChannelsSpreader::ChannelsSpreader(int _a, int _w, int _h)
{
    this->expectedChannelsAmount = _a;
    this->expectedChannelsWidth = _w;
    this->expectedChannelsHeight = _h;
}

void ChannelsSpreader::initialize(NeurologicalComponentInitializationInfo info)
{
    // empty
}

int ChannelsSpreader::getOutputSize()
{
	return this->expectedChannelsAmount*this->expectedChannelsWidth*this->expectedChannelsHeight;
}

void ChannelsSpreader::forwardPass(const neurologicalListValues& _input, neurologicalTensorChannels& _output, const string& sampleIndex)
{
    for (int i = 0; i < this->expectedChannelsAmount; i++) {
        _output[i] = neurologicalMatrixValues(this->expectedChannelsHeight);
        for (int _y = 0; _y < this->expectedChannelsHeight; _y++) {
            _output[i][_y] = neurologicalListValues(this->expectedChannelsWidth);
            for (int _x = 0; _x < this->expectedChannelsWidth; _x++) {
                _output[i][_y][_x] = _input[_x + (_y*this->expectedChannelsWidth) + (i*this->expectedChannelsWidth*this->expectedChannelsHeight)];
            }
        }
    }

    // cout << "spreaded" << endl;
}

void ChannelsSpreader::backPropagation(const neurologicalListValues& _propagation, neurologicalTensorChannels& _inputPropagation, const string& sampleIndex)
{
    // EMPTY -- again.. does NOT support back propagation
}

void ChannelsSpreader::applyDerivatives(derivativesApplyingInfo applyingInfo)
{
    // EMPTY
}