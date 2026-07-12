// Fill out your copyright notice in the Description page of Project Settings.


#include "SoftPlus.h"

SoftPlus::SoftPlus()
{

}

void SoftPlus::__activation(const neurologicalListValues& _listInput, neurologicalListValues& _listOutput)
{
	for (int i = 0; i < _listInput.size(); i++) {
		_listOutput[i] = log(1.0 + exp(_listInput[i]));
	}
}

void SoftPlus::__gradient(const neurologicalListValues& _listPropagation, const neurologicalListValues& _listInput, const neurologicalListValues& _listOutput, neurologicalListValues& _listInputPropagation)
{
	for (int i = 0; i < _listPropagation.size(); i++) {
		_listInputPropagation[i] = (1.0 / (1.0 + exp(-_listInput[i]))) * _listPropagation[i];
	}
}

void SoftPlus::activation(const neurologicalListValues& _listInput, neurologicalListValues& _listOutput)
{
	this->__activation(_listInput, _listOutput);
}

void SoftPlus::gradient(const neurologicalListValues& _listPropagation, const neurologicalListValues& _listInput, const neurologicalListValues& _listOutput, neurologicalListValues& _listInputPropagation)
{
	this->__gradient(_listPropagation, _listInput, _listOutput, _listInputPropagation);
}