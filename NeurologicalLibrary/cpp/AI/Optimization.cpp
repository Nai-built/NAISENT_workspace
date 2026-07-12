// Fill out your copyright notice in the Description page of Project Settings.


#include "Optimization.h"

#include <iostream>
#include <variant>
#include <vector>
#include <optional>
#include <memory>

using namespace std;

NONE_parameterOptimizationInfo::NONE_parameterOptimizationInfo()
{
	// its literally empty
}

// --- ADAM OPTIMIZER ---

ADAM_parameterOptimizationInfo::ADAM_parameterOptimizationInfo()
{
	v = 0;
	m = 0;

	t = 0;
}

//ADAM_Optimizer::ADAM_Optimizer()
//{
//	this->epsilon = 0;
//	this->beta1 = 0;
//	this->beta2 = 0;
//}

ADAM_Optimizer::ADAM_Optimizer(float _epsilon, float _beta1, float _beta2)
{
	this->epsilon = _epsilon;
	this->beta1 = _beta1;
	this->beta2 = _beta2;
}

parameterOptimizationInfo ADAM_Optimizer::initializeParameter()
{
	return ADAM_parameterOptimizationInfo();
}

neurologicalValue ADAM_Optimizer::optimizeParameterDerivative(derivativesApplyingInfo applyingInfo, parameterOptimizationInfo& parameterInfo, neurologicalValue derivative)
{
	ADAM_parameterOptimizationInfo& _parameterInfo = get<ADAM_parameterOptimizationInfo>(parameterInfo);

	_parameterInfo.t++;

	_parameterInfo.m = this->beta1 * _parameterInfo.m + (1 - this->beta1) * derivative;
	_parameterInfo.v = this->beta2 * _parameterInfo.v + (1 - this->beta2) * (derivative * derivative);

	neurologicalValue m_hat = _parameterInfo.m / (1 - pow(this->beta1, _parameterInfo.t));
	neurologicalValue v_hat = _parameterInfo.v / (1 - pow(this->beta2, _parameterInfo.t));

	neurologicalValue optimizedDerivative = (applyingInfo.learningRate * m_hat / (pow(v_hat, 0.5) + this->epsilon));

	return optimizedDerivative;
}

// ^-- CLIPPING METHODS --^

// -- NORM --

NORM_ClippingMethod::NORM_ClippingMethod(neurologicalValue _clip_value)
{
	this->clip_value = _clip_value;
}

neurologicalValue NORM_ClippingMethod::clip(neurologicalValue derivative)
{
	neurologicalValue norm = sqrt(this->poweredDerivativesSum);

	if (norm > this->clip_value)
	{
		return derivative * (this->clip_value / norm);
	}
	return derivative;
}