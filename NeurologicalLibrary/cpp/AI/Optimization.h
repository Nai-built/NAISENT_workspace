// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "Neurological.h"

#include <iostream>
#include <variant>
#include <vector>
#include <optional>
#include <memory>

using namespace std;

struct NONE_parameterOptimizationInfo {
	NONE_parameterOptimizationInfo();
};

struct ADAM_parameterOptimizationInfo {
	neurologicalValue v;
	neurologicalValue m;

	int t;

	ADAM_parameterOptimizationInfo();
};

using parameterOptimizationInfo = variant<NONE_parameterOptimizationInfo, ADAM_parameterOptimizationInfo>;

/**
 * 
 */

class BaseOptimizer
{
public:
	virtual parameterOptimizationInfo initializeParameter() = 0;

	virtual neurologicalValue optimizeParameterDerivative(derivativesApplyingInfo applyingInfo, parameterOptimizationInfo& _parameterInfo, neurologicalValue derivative) = 0;
};

class ADAM_Optimizer : public BaseOptimizer
{
public:
	float epsilon;
	float beta1;
	float beta2;

	ADAM_Optimizer(float _epsilon, float _beta1, float _beta2);

	parameterOptimizationInfo initializeParameter() override;

	neurologicalValue optimizeParameterDerivative(derivativesApplyingInfo applyingInfo, parameterOptimizationInfo& _parameterInfo, neurologicalValue derivative) override;
};

using OptimizerPTR = shared_ptr<BaseOptimizer>;
using OptimizerVariable = optional<OptimizerPTR>;

class BaseClippingMethod
{
public:
	virtual neurologicalValue clip(neurologicalValue derivative) = 0;
};

class NORM_ClippingMethod : public BaseClippingMethod
{
protected:
	neurologicalValue clip_value;
public:
	neurologicalValue poweredDerivativesSum = 0;

	NORM_ClippingMethod(neurologicalValue _clip_value);

	neurologicalValue clip(neurologicalValue derivative) override;
};

using ClippingMethodPTR = shared_ptr<BaseClippingMethod>;
using ClippingMethodVariable = optional<ClippingMethodPTR>;