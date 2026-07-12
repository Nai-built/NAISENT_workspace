// Fill out your copyright notice in the Description page of Project Settings.


#include "Tanh.h"

Tanh::Tanh()
{

}

neurologicalValue Tanh::__derivative(const neurologicalValue& outputValue)
{
	return ((double)1) - pow(outputValue, 2);
}

void Tanh::__activation(const neurologicalListValues& _listInput, neurologicalListValues& _listOutput)
{
	for (int i = 0; i < _listInput.size(); i++) {
		_listOutput[i] = tanh(_listInput[i]);
	}
}

void Tanh::__gradient(const neurologicalListValues& _listPropagation, const neurologicalListValues& _listInput, const neurologicalListValues& _listOutput, neurologicalListValues& _listInputPropagation)
{
	for (int i = 0; i < _listPropagation.size(); i++) {
		// _gradient[i] = (((double)1) - pow(tanh(_listInput[i]), 2))*_listPropagation[i];
		_listInputPropagation[i] = Tanh::__derivative(tanh(_listInput[i]))*_listPropagation[i];
	}
}

void Tanh::activation(const neurologicalListValues& _listInput, neurologicalListValues& _listOutput)
{
	this->__activation(_listInput, _listOutput);
}

void Tanh::gradient(const neurologicalListValues& _listPropagation, const neurologicalListValues& _listInput, const neurologicalListValues& _listOutput, neurologicalListValues& _listInputPropagation)
{
	this->__gradient(_listPropagation, _listInput, _listOutput, _listInputPropagation);
}