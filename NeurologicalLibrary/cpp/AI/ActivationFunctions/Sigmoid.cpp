// Fill out your copyright notice in the Description page of Project Settings.

#include <math.h>

#include "Sigmoid.h"

Sigmoid::Sigmoid()
{

}

neurologicalValue Sigmoid::__result(const neurologicalValue& inputValue) {
    return ((double)1.0) / (((double)1.0) + std::exp(-inputValue));
}
neurologicalValue Sigmoid::__derivative(const neurologicalValue& outputValue) {
    return (outputValue*(((double)1)-outputValue));
}

void Sigmoid::__activation(const neurologicalListValues& _listInput, neurologicalListValues& _listOutput)
{
	for (int i = 0; i < _listInput.size(); i++) {
		_listOutput[i] = Sigmoid::__result(_listInput[i]);
	}
}

void Sigmoid::__gradient(const neurologicalListValues& _listPropagation, const neurologicalListValues& _listInput, const neurologicalListValues& _listOutput, neurologicalListValues& _listInputPropagation)
{
	for (int i = 0; i < _listPropagation.size(); i++) {
        neurologicalValue _sigmoidValue = Sigmoid::__result(_listInput[i]);
		_listInputPropagation[i] = Sigmoid::__derivative(_sigmoidValue)*_listPropagation[i];
	}
}

void Sigmoid::activation(const neurologicalListValues& _listInput, neurologicalListValues& _listOutput)
{
	this->__activation(_listInput, _listOutput);
}

void Sigmoid::gradient(const neurologicalListValues& _listPropagation, const neurologicalListValues& _listInput, const neurologicalListValues& _listOutput, neurologicalListValues& _listInputPropagation)
{
	this->__gradient(_listPropagation, _listInput, _listOutput, _listInputPropagation);
}