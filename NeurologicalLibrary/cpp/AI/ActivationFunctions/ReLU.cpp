// Fill out your copyright notice in the Description page of Project Settings.


#include "ReLU.h"

ReLU::ReLU()
{
	this->leakage = 0;
}
ReLU::ReLU(float _leakage)
{
	this->leakage = _leakage;
}

void ReLU::__activation(float leakage, const neurologicalListValues& _listInput, neurologicalListValues& _listOutput)
{
	for (int i = 0; i < _listInput.size(); i++) {
		if (_listInput[i] > 0) {
			_listOutput[i] = _listInput[i];
		}
		else {
			_listOutput[i] = _listInput[i] * leakage;
		}
	}
}

void ReLU::__gradient(float leakage, const neurologicalListValues& _listPropagation, const neurologicalListValues& _listInput, const neurologicalListValues& _listOutput, neurologicalListValues& _listInputPropagation)
{
	for (int i = 0; i < _listPropagation.size(); i++) {
		if (_listInput[i] > 0) {
			_listInputPropagation[i] = _listPropagation[i];
		}
		else {
			_listInputPropagation[i] = _listPropagation[i] * leakage;
		}
	}
}

void ReLU::activation(const neurologicalListValues& _listInput, neurologicalListValues& _listOutput)
{
	this->__activation(this->leakage, _listInput, _listOutput);
}

void ReLU::gradient(const neurologicalListValues& _listPropagation, const neurologicalListValues& _listInput, const neurologicalListValues& _listOutput, neurologicalListValues& _listInputPropagation)
{
	this->__gradient(this->leakage, _listPropagation, _listInput, _listOutput, _listInputPropagation);
}