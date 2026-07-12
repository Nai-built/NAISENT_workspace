// // Fill out your copyright notice in the Description page of Project Settings.


// #include "MultiLayer.h"

// #include <vector>
// #include <initializer_list>
// #include <variant>
// #include <optional>

// // CURRENTLY ONLY SUPPORTING DENSE LAYERS!!

// using namespace std;

// MultiLayer::MultiLayer(vector<INeurologicalComponentPTR> _branches)
// {
// 	branches = _branches;
// }

// void MultiLayer::initialize(NeurologicalComponentInitializationInfo info)
// {
// 	this->optimizer = info.optimizer;

// 	for (int i = 0; i < branches.size(); i++) {
// 		INeurologicalComponentPTR element = branches[i];
// 		element->initialize(info);
// 	}
// }

// int MultiLayer::getOutputSize()
// {
// 	int outputSize = 0;

// 	for (int i = 0; i < this->branches.size(); i++)
// 	{
// 		outputSize += this->branches[i]->getOutputSize();
// 	}

// 	return outputSize;
// }

// NeurologicalPassingValues MultiLayer::forwardPass(NeurologicalPassingValues _input, const string& sampleIndex)
// {
// 	neurologicalListValues result;
// 	vector<neurologicalListValues> branchesResults;
// 	for (int i = 0; i < branches.size(); i++)
// 	{
// 		INeurologicalComponentPTR branch = branches[i];
// 		neurologicalListValues branchOutput = get<neurologicalListValues>(branch->forwardPass(_input, sampleIndex));
// 		branchesResults.push_back(branchOutput);
// 	}
// 	result = joinLists<neurologicalValue>(branchesResults);

// 	return result;
// }
// NeurologicalPassingValues MultiLayer::backPropagation(NeurologicalPassingValues _propagation, const string& sampleIndex)
// {
// 	neurologicalListValues propagation = get<neurologicalListValues>(_propagation);
// 	vector<neurologicalListValues> branchesInputPropagations;

// 	int _propagationIndex = 0;
// 	for (int i = 0; i < this->branches.size(); i++)
// 	{
// 		INeurologicalComponentPTR branch = branches[i];
// 		neurologicalListValues branchPropagation;
// 		//cout << to_string(this->branches[i]->getOutputSize()) + " OUTPUT SIZE\n";
// 		for (int j = 0; j < this->branches[i]->getOutputSize(); j++)
// 		{
// 			branchPropagation.push_back(propagation[_propagationIndex]);

// 			_propagationIndex++;
// 		}

// 		neurologicalListValues branchInputPropagation = get<neurologicalListValues>(branch->backPropagation(branchPropagation, sampleIndex));
// 		branchesInputPropagations.push_back(branchInputPropagation);
// 	}

// 	neurologicalListValues inputPropagation = addListsValues<neurologicalValue>(branchesInputPropagations);

// 	return inputPropagation;
// }

// void MultiLayer::applyDerivatives(derivativesApplyingInfo applyingInfo)
// {
// 	for (int i = 0; i < branches.size(); i++) {
// 		INeurologicalComponentPTR element = branches[i];
// 		element->applyDerivatives(applyingInfo);
// 	}
// }

// //NeurologicalComponentsChain::~NeurologicalComponentsChain()
// //{
// //	/*for (INeurologicalComponent* _component : this->chain)
// //	{
// //		delete _component;
// //	}*/
// //}