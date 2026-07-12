// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "Extra.h"

#include <iostream>
#include <variant>
#include <vector>
#include <unordered_map>
#include <optional>
#include <memory>

using namespace std;

typedef float neurologicalValue;

typedef vector<neurologicalValue> neurologicalListValues;
typedef vector<vector<neurologicalValue>> neurologicalMatrixValues;
typedef vector<neurologicalMatrixValues> neurologicalTensorChannels;

struct neurologicalGraphNode {
	neurologicalListValues values;
	vector<string> bridges;
};

typedef unordered_map<string, neurologicalGraphNode> neurologicalGraphNodes;

// using NeurologicalPassingValues = variant<neurologicalListValues, neurologicalMatrixValues, convolutionalChannelsValues, neurologicalGraphNodes>;

/**
 * 
 */

struct derivativesApplyingInfo {
public:
	float learningRate;
	int averagingDivisor;
};