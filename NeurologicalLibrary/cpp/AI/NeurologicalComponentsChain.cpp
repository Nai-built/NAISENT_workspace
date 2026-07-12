// Fill out your copyright notice in the Description page of Project Settings.


#include "NeurologicalComponentsChain.h"

#include "BaseActivationFunction.h"
#include "DenseLayer.h"
#include "LSTM_RecursiveLayer.h"
#include "ConvolutionalLayer.h"
#include "ConvolutionalPool.h"
#include "ChannelsFlattener.h"
#include "ChannelsSpreader.h"

#include <vector>
#include <variant>
#include <optional>

using namespace std;

template<typename T>
optional<shared_ptr<T>> CheckNeurologicalComponent(INeurologicalComponentPTR component) {
    shared_ptr<T> _cast = dynamic_pointer_cast<T>(component);

    if (_cast == nullptr) {
        return nullopt;
    }

    return _cast;
}

void NeurologicalComponentsChain::buildComponentForwardPass(vector<componentCallback>& program, const int& _index, INeurologicalComponentPTR component)
{
	int targetIndex = _index+1;

	if (auto denseLayer = CheckNeurologicalComponent<DenseLayer>(component)) {
		program.push_back([&
			, denseLayer, _index, targetIndex](string sampleIndex) {
			this->listsBuffer[sampleIndex][targetIndex] = neurologicalListValues(denseLayer.value()->neuronsSize);
			denseLayer.value()->forwardPass(this->listsBuffer[sampleIndex][_index]
			, this->listsBuffer[sampleIndex][targetIndex], sampleIndex);
		});
	} else if (auto LSTM_recursiveLayer = CheckNeurologicalComponent<LSTM_RecursiveLayer>(component)) {
		program.push_back([&
			, LSTM_recursiveLayer, _index, targetIndex](string sampleIndex) {
			this->listsBuffer[sampleIndex][targetIndex] = neurologicalListValues(LSTM_recursiveLayer.value()->outputSize);
			LSTM_recursiveLayer.value()->forwardPass(this->listsBuffer[sampleIndex][_index]
			, this->listsBuffer[sampleIndex][targetIndex], sampleIndex);
		});
	} else if (auto convolutionalLayer = CheckNeurologicalComponent<ConvolutionalLayer>(component)) {
		program.push_back([&
			, convolutionalLayer, _index, targetIndex](string sampleIndex) {
			this->tensorsBuffer[sampleIndex][targetIndex] = neurologicalTensorChannels(convolutionalLayer.value()->filterChannelsSize);
			convolutionalLayer.value()->forwardPass(this->tensorsBuffer[sampleIndex][_index]
			, this->tensorsBuffer[sampleIndex][targetIndex], sampleIndex);
		});
	} else if (auto convolutionalPool = CheckNeurologicalComponent<ConvolutionalPool>(component)) {
		program.push_back([&
			, convolutionalPool, _index, targetIndex](string sampleIndex) {
			this->tensorsBuffer[sampleIndex][targetIndex] = neurologicalTensorChannels(this->tensorsBuffer[sampleIndex][_index].size());
			convolutionalPool.value()->forwardPass(this->tensorsBuffer[sampleIndex][_index]
			, this->tensorsBuffer[sampleIndex][targetIndex], sampleIndex);
		});
	} else if (auto channelsFlattener = CheckNeurologicalComponent<ChannelsFlattener>(component)) {
		program.push_back([&
			, channelsFlattener, _index, targetIndex](string sampleIndex) {
			this->listsBuffer[sampleIndex][targetIndex] = neurologicalListValues();
			channelsFlattener.value()->forwardPass(this->tensorsBuffer[sampleIndex][_index]
			, this->listsBuffer[sampleIndex][targetIndex], sampleIndex);
		});
	} else if (auto channelsSpreader = CheckNeurologicalComponent<ChannelsSpreader>(component)) {
		program.push_back([&
			, channelsSpreader, _index, targetIndex](string sampleIndex) {
			this->tensorsBuffer[sampleIndex][targetIndex] = neurologicalTensorChannels(channelsSpreader.value()->expectedChannelsAmount);
			channelsSpreader.value()->forwardPass(this->listsBuffer[sampleIndex][_index]
			, this->tensorsBuffer[sampleIndex][targetIndex], sampleIndex);
		});
	} else if (auto activationFunction = CheckNeurologicalComponent<BaseActivationFunction>(component)) {
		program.push_back([&
			, activationFunction, _index, targetIndex](string sampleIndex) {
			if (this->listsBuffer[sampleIndex].contains(_index)) {
				this->listsBuffer[sampleIndex][targetIndex] = neurologicalListValues(this->listsBuffer[sampleIndex][_index].size());
				activationFunction.value()->forwardPass__list(this->listsBuffer[sampleIndex][_index]
				, this->listsBuffer[sampleIndex][targetIndex], sampleIndex);
			}
			else if (this->tensorsBuffer[sampleIndex].contains(_index)) {
				this->tensorsBuffer[sampleIndex][targetIndex] = neurologicalTensorChannels(this->tensorsBuffer[sampleIndex][_index].size());
				activationFunction.value()->forwardPass__tensor(this->tensorsBuffer[sampleIndex][_index]
				, this->tensorsBuffer[sampleIndex][targetIndex], sampleIndex);
			}
		});
	}

	// cant have a chain inside another chain directly!
}

void NeurologicalComponentsChain::buildComponentBackPropagation(vector<componentCallback>& program, const int& _index, INeurologicalComponentPTR component)
{
	int targetIndex = _index-1;

	if (auto denseLayer = CheckNeurologicalComponent<DenseLayer>(component)) {
		program.push_back([&
			, denseLayer, _index, targetIndex](string sampleIndex) {
			this->listsPropagation[sampleIndex][targetIndex] = neurologicalListValues(denseLayer.value()->inputSize);
			denseLayer.value()->backPropagation(this->listsPropagation[sampleIndex][_index]
			, this->listsPropagation[sampleIndex][targetIndex], sampleIndex);
		});
	} else if (auto LSTM_recursiveLayer = CheckNeurologicalComponent<LSTM_RecursiveLayer>(component)) {
		program.push_back([&
			, LSTM_recursiveLayer, _index, targetIndex](string sampleIndex) {
			this->listsPropagation[sampleIndex][targetIndex] = neurologicalListValues(LSTM_recursiveLayer.value()->inputSize);
			LSTM_recursiveLayer.value()->backPropagation(this->listsPropagation[sampleIndex][_index]
			, this->listsPropagation[sampleIndex][targetIndex], sampleIndex);
		});
	} else if (auto convolutionalLayer = CheckNeurologicalComponent<ConvolutionalLayer>(component)) {
		program.push_back([&
			, convolutionalLayer, _index, targetIndex](string sampleIndex) {
			this->tensorsPropagation[sampleIndex][targetIndex] = neurologicalTensorChannels(convolutionalLayer.value()->inputChannelsSize);
			convolutionalLayer.value()->backPropagation(this->tensorsPropagation[sampleIndex][_index]
			, this->tensorsPropagation[sampleIndex][targetIndex], sampleIndex);
		});
	} else if (auto convolutionalPool = CheckNeurologicalComponent<ConvolutionalPool>(component)) {
		program.push_back([&
			, convolutionalPool, _index, targetIndex](string sampleIndex) {
			this->tensorsPropagation[sampleIndex][targetIndex] = neurologicalTensorChannels(this->tensorsPropagation[sampleIndex][_index].size());
			convolutionalPool.value()->backPropagation(this->tensorsPropagation[sampleIndex][_index]
			, this->tensorsPropagation[sampleIndex][targetIndex], sampleIndex);
		});
	} else if (auto channelsFlattener = CheckNeurologicalComponent<ChannelsFlattener>(component)) {
		program.push_back([&
			, channelsFlattener, _index, targetIndex](string sampleIndex) {
			this->tensorsPropagation[sampleIndex][targetIndex] = neurologicalTensorChannels();
			channelsFlattener.value()->backPropagation(this->listsPropagation[sampleIndex][_index]
				, this->tensorsPropagation[sampleIndex][targetIndex], sampleIndex);
		});
	} else if (auto channelsSpreader = CheckNeurologicalComponent<ChannelsSpreader>(component)) {
		// channels spreader does NOT support back propagation
		program.push_back([&
			, channelsSpreader, _index, targetIndex](string sampleIndex) {
				this->listsPropagation[sampleIndex][targetIndex] = neurologicalListValues();
				// cout << "back propagation is not supported in spreaders" << endl;
		});
	} else if (auto activationFunction = CheckNeurologicalComponent<BaseActivationFunction>(component)) {
		program.push_back([&
			, activationFunction, _index, targetIndex](string sampleIndex) {
			if (this->listsPropagation[sampleIndex].contains(_index)) {
				this->listsPropagation[sampleIndex][targetIndex] = neurologicalListValues(this->listsPropagation[sampleIndex][_index].size());
				activationFunction.value()->backPropagation__list(this->listsPropagation[sampleIndex][_index]
				, this->listsPropagation[sampleIndex][targetIndex], sampleIndex);
			}
			else if (this->tensorsPropagation[sampleIndex].contains(_index)) {
				this->tensorsPropagation[sampleIndex][targetIndex] = neurologicalTensorChannels(this->tensorsPropagation[sampleIndex][_index].size());
				activationFunction.value()->backPropagation__tensor(this->tensorsPropagation[sampleIndex][_index]
				, this->tensorsPropagation[sampleIndex][targetIndex], sampleIndex);
			}
		});

	}
	// cant have a chain inside another chain directly!
}

NeurologicalComponentsChain::NeurologicalComponentsChain(vector<INeurologicalComponentPTR> _chain)
{
	chain = _chain;
}

void NeurologicalComponentsChain::initialize(NeurologicalComponentInitializationInfo info)
{
	this->optimizer = info.optimizer;

	for (int i = 0; i < chain.size(); i++) {
		INeurologicalComponentPTR element = chain[i];
		element->initialize(info);

		this->buildComponentForwardPass(this->forwardPassProgram, i, element);
		this->buildComponentBackPropagation(this->backPropagationProgram, i, element);
	}
}

int NeurologicalComponentsChain::getOutputSize()
{
	INeurologicalComponentPTR element;
	for (int i = this->chain.size() - 1; i >= 0; i--) {
		element = this->chain[i];
		if (!dynamic_pointer_cast<BaseActivationFunction>(element)) {
			break;
		}
	}

	return element->getOutputSize();
}

NeurologicalPassingValues NeurologicalComponentsChain::forwardPass(NeurologicalPassingValues _input, const string& sampleIndex)
{
	// NeurologicalPassingValues result = _input;
	// for (int i = 0; i < chain.size(); i++) {
	// 	INeurologicalComponentPTR element = chain[i];
	// 	result = element->forwardPass(result, sampleIndex);
	// }
	// return result;

	this->listsBuffer[sampleIndex] = unordered_map<int, neurologicalListValues>();
	this->tensorsBuffer[sampleIndex] = unordered_map<int, neurologicalTensorChannels>();

	// set input properly
	if (holds_alternative<neurologicalListValues>(_input)) {
		this->listsBuffer[sampleIndex][0] = get<neurologicalListValues>(_input);
	} else if (holds_alternative<neurologicalTensorChannels>(_input)) {
		this->tensorsBuffer[sampleIndex][0] = get<neurologicalTensorChannels>(_input);
	}

	// pass through chain properly
	for (int i = 0; i < this->forwardPassProgram.size(); i++) {
		this->forwardPassProgram[i](sampleIndex);
	}

	// get output index
	int outputIndex = this->chain.size();

	// extract output properly
	if (this->listsBuffer[sampleIndex].contains(outputIndex)) {
		return this->listsBuffer[sampleIndex][outputIndex];
	} else if (this->tensorsBuffer[sampleIndex].contains(outputIndex)) {
		return this->tensorsBuffer[sampleIndex][outputIndex];
	}
}
NeurologicalPassingValues NeurologicalComponentsChain::backPropagation(NeurologicalPassingValues _propagation, const string& sampleIndex)
{
	// NeurologicalPassingValues propagation = _propagation;

	// for (int i = chain.size()-1; i >= 0; i--) {
	// 	INeurologicalComponentPTR element = chain[i];
	// 	propagation = element->backPropagation(propagation, sampleIndex);
	// }
	// return propagation;

	this->listsPropagation[sampleIndex] = unordered_map<int, neurologicalListValues>();
	this->tensorsPropagation[sampleIndex] = unordered_map<int, neurologicalTensorChannels>();

	// set propagation properly
	if (holds_alternative<neurologicalListValues>(_propagation)) {
		this->listsPropagation[sampleIndex][chain.size()-1] = get<neurologicalListValues>(_propagation);
	} else if (holds_alternative<neurologicalTensorChannels>(_propagation)) {
		this->tensorsPropagation[sampleIndex][chain.size()-1] = get<neurologicalTensorChannels>(_propagation);
	}

	// propagate through chain properly
	for (int i = backPropagationProgram.size()-1; i >= 0; i--) {
		this->backPropagationProgram[i](sampleIndex);
	}

	// get final propagation index
	int finalPropagationIndex = -1;

	// extract output properly
	if (this->listsPropagation[sampleIndex].contains(finalPropagationIndex)) {
		return this->listsPropagation[sampleIndex][finalPropagationIndex];
	} else if (this->tensorsPropagation[sampleIndex].contains(finalPropagationIndex)) {
		return this->tensorsPropagation[sampleIndex][finalPropagationIndex];
	}
}

void NeurologicalComponentsChain::applyDerivatives(derivativesApplyingInfo applyingInfo)
{
	for (int i = 0; i < chain.size(); i++) {
		INeurologicalComponentPTR element = chain[i];
		element->applyDerivatives(applyingInfo);
	}

	this->listsBuffer.clear();
	this->tensorsBuffer.clear();
	this->listsPropagation.clear();
	this->tensorsPropagation.clear();
}

//NeurologicalComponentsChain::~NeurologicalComponentsChain()
//{
//	/*for (INeurologicalComponent* _component : this->chain)
//	{
//		delete _component;
//	}*/
//}