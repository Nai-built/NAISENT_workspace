// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

//#include "CoreMinimal.h"

#include "../BaseActivationFunction.h"

/**
 * 
 */
class Softmax : public BaseActivationFunction
{
public:
	Softmax();
	Softmax(bool ignoreJacobian);

	bool ignoreJacobian;

	static void __activation(const neurologicalListValues& _listInput, neurologicalListValues& _listOutput);
	static void _jacobian(const neurologicalListValues& _softmax, vector<neurologicalListValues>& _jacobianResult);
	static void __gradient(bool ignoreJacobian, const neurologicalListValues& _listPropagation, const neurologicalListValues& _listInput, const neurologicalListValues& _listOutput, neurologicalListValues& _listInputPropagation);

	void activation(const neurologicalListValues& _listInput, neurologicalListValues& _listOutput) override;
	void gradient(const neurologicalListValues& _listPropagation, const neurologicalListValues& _listInput, const neurologicalListValues& _listOutput, neurologicalListValues& _listInputPropagation) override;
};
