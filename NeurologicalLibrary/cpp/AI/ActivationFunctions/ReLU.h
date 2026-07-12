// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

//#include "CoreMinimal.h"

#include "../BaseActivationFunction.h"

/**
 * 
 */
class ReLU : public BaseActivationFunction
{
private:
	float leakage;
public:
	ReLU();
	ReLU(float _leakage);

	static void __activation(float leakage, const neurologicalListValues& _listInput, neurologicalListValues& _listOutput);
	static void __gradient(float leakage, const neurologicalListValues& _listPropagation, const neurologicalListValues& _listInput, const neurologicalListValues& _listOutput, neurologicalListValues& _listInputPropagation);

	void activation(const neurologicalListValues& _listInput, neurologicalListValues& _listOutput) override;
	void gradient(const neurologicalListValues& _listPropagation, const neurologicalListValues& _listInput, const neurologicalListValues& _listOutput, neurologicalListValues& _listInputPropagation) override;
};
