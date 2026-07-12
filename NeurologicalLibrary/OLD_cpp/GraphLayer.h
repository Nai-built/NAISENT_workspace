// // Fill out your copyright notice in the Description page of Project Settings.

// #pragma once

// #include "NeurologicalComponent.h"
// #include "DenseLayer.h"

// #include <vector>
// #include <optional>
// #include <unordered_map>

// using namespace std;

// /**
//  *
//  */

// class GraphLayer : public INeurologicalComponent, public enable_shared_from_this<GraphLayer>
// {
// public:
// 	INeurologicalComponentPTR processingLayer;

// 	optional<INeurologicalComponentPTR> messagingLayer;
// 	optional<INeurologicalComponentPTR> attentionLayer;

// 	unordered_map<string, unordered_map<string, unordered_map<string, neurologicalValue>>> recordedAlphas;
// 	unordered_map<string, unordered_map<string, unordered_map<string, neurologicalListValues>>> recordedMessages;

// 	unordered_map<string, neurologicalGraphNodes> recordedOutputs;

// 	int inputSize;
// 	int outputSize;

// 	optional<string> selectedNode;

// 	GraphLayer(int inputSize, int outputSize, bool messaging = false, bool attention = false);

// 	void initialize(NeurologicalComponentInitializationInfo info) override;

// 	int getOutputSize() override;

// 	NeurologicalPassingValues forwardPass(NeurologicalPassingValues _input, const string& sampleIndex) override;
// 	NeurologicalPassingValues backPropagation(NeurologicalPassingValues _propagation, const string& sampleIndex) override;
// 	void applyDerivatives(derivativesApplyingInfo applyingInfo) override;

// 	shared_ptr<GraphLayer> selectNode(string nodeName);
// };
