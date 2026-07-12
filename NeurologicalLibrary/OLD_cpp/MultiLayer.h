// // Fill out your copyright notice in the Description page of Project Settings.

// #pragma once

// #include "NeurologicalComponent.h"

// #include <variant>
// #include <vector>
// #include <optional>
// #include <memory>

// using namespace std;

// /**
//  * 
//  */
// class MultiLayer : public INeurologicalComponent
// {
// public:
// 	vector<INeurologicalComponentPTR> branches;

// 	MultiLayer(vector<INeurologicalComponentPTR> _branches);

// 	void initialize(NeurologicalComponentInitializationInfo info) override;

// 	int getOutputSize() override;

// 	NeurologicalPassingValues forwardPass(NeurologicalPassingValues _input, const string& sampleIndex) override;
// 	NeurologicalPassingValues backPropagation(NeurologicalPassingValues _propagation, const string& sampleIndex) override;
// 	void applyDerivatives(derivativesApplyingInfo applyingInfo) override;
// };
