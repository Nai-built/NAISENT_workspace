// Fill out your copyright notice in the Description page of Project Settings.

#include "NeurologicalComponent.h"

#include <mutex>

using namespace std;

//union _t {
//	char c[10];
//	int i[100];
//	bool isB[0];
//
//	bool getItQuestionMark() {
//		return true;
//	}[5];
//};
//
//class ITest
//{
//public:
//	ITest() {
//	};
//
//	virtual char tttttt(int _input) = 0;
//};
//
//class Test1 : public ITest
//{
//public:
//	Test1() {
//
//	};
//
//	char tttttt(int _input) override {
//
//	};
//};

NeurologicalParameter::NeurologicalParameter() {
	/*_t u = _t();

	u.getItQuestionMark();

	ITest* t = new Test1();*/

	//this->_mutex = make_shared<mutex>();

	value = 0;
	calculatedDerivative = 0;

	this->optimizationInfo = NONE_parameterOptimizationInfo();
}

NeurologicalParameter::NeurologicalParameter(neurologicalValue initialValue, parameterOptimizationInfo optimizationInfo = NONE_parameterOptimizationInfo()) {
	//this->_mutex = make_shared<mutex>();

	value = initialValue;
	calculatedDerivative = 0;

	this->optimizationInfo = optimizationInfo;
}
NeurologicalParameter::NeurologicalParameter(neurologicalValue initialValue) {
	value = initialValue;
	calculatedDerivative = 0;
}

parameterOptimizationInfo NeurologicalParameter::getOptimizationInfo()
{
	return this->optimizationInfo;
}

void NeurologicalParameter::addCalculatedDerivative(neurologicalValue derivative)
{
	//lock_guard<mutex> _lock(*this->_mutex);

	this->calculatedDerivative += derivative;
}
void NeurologicalParameter::applyCalculatedDerivatives(derivativesApplyingInfo applyingInfo, OptimizerVariable optimizer, ClippingMethodVariable clippingMethod) {
	//lock_guard<mutex> _lock(*this->_mutex);

	neurologicalValue derivative = this->calculatedDerivative / applyingInfo.averagingDivisor;

	if (clippingMethod.has_value()) {
		derivative = clippingMethod.value()->clip(derivative);
	}

	if (!optimizer.has_value()) {
		this->value -= (derivative * applyingInfo.learningRate);
	}
	else {
		this->value -= optimizer.value()->optimizeParameterDerivative(applyingInfo, this->optimizationInfo, derivative);
	}

	this->calculatedDerivative = 0;
}

neurologicalValue NeurologicalParameter::checkCalculatedDerivative()
{
	return this->calculatedDerivative;
}