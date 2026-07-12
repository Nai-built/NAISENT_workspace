// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

//#include "CoreMinimal.h"

#include "../BaseActivationFunction.h"

/**
 *
 */
class Sigmoid : public BaseActivationFunction
{
public:
	Sigmoid();

	static neurologicalValue __result(const neurologicalValue& inputValue);
	static neurologicalValue __derivative(const neurologicalValue& outputValue);

	static void __activation(const neurologicalListValues& _listInput, neurologicalListValues& _listOutput);
	static void __gradient(const neurologicalListValues& _listPropagation, const neurologicalListValues& _listInput, const neurologicalListValues& _listOutput, neurologicalListValues& _listInputPropagation);

	void activation(const neurologicalListValues& _listInput, neurologicalListValues& _listOutput) override;
	void gradient(const neurologicalListValues& _listPropagation, const neurologicalListValues& _listInput, const neurologicalListValues& _listOutput, neurologicalListValues& _listInputPropagation) override;
};
