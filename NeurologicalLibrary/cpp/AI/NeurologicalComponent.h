// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "Optimization.h"
#include "Neurological.h"

#include <iostream>
#include <variant>
#include <vector>
#include <optional>
#include <memory>
#include <mutex>
#include <string>

using namespace std;

/**
 *
 */

struct NeurologicalParameter
{
private:
	//shared_ptr<mutex> _mutex;

	neurologicalValue calculatedDerivative;

public:
	neurologicalValue value;
	parameterOptimizationInfo optimizationInfo;

	NeurologicalParameter();
	NeurologicalParameter(neurologicalValue initialValue);
	NeurologicalParameter(neurologicalValue initialValue, parameterOptimizationInfo optimizationInfo);

	parameterOptimizationInfo getOptimizationInfo();

	void addCalculatedDerivative(neurologicalValue derivative);
	void applyCalculatedDerivatives(derivativesApplyingInfo applyingInfo, OptimizerVariable optimizer = nullopt, ClippingMethodVariable clippingMethod = nullopt);

	neurologicalValue checkCalculatedDerivative();
};

struct NeurologicalComponentInitializationInfo {
	OptimizerVariable optimizer = nullopt;
	ClippingMethodVariable clippingMethod = nullopt;
};

// INTERFACE
class INeurologicalComponent
{
protected:
	// mutex input_mutex;
	// mutex output_mutex;
	// mutex derivative_mutex;
public:
	OptimizerVariable optimizer;
	ClippingMethodVariable clippingMethod;

	virtual void initialize(NeurologicalComponentInitializationInfo info) = 0;

	virtual int getOutputSize() = 0;

	virtual void applyDerivatives(derivativesApplyingInfo applyingInfo) = 0;
};

using INeurologicalComponentPTR = shared_ptr<INeurologicalComponent>;