// #include "GraphLayer.h"

// #include <iostream>
// #include <utility>
// #include <string>

// using namespace std;

// GraphLayer::GraphLayer(int _inputSize, int _outputSize, bool messaging, bool attention)
// {
// 	this->inputSize = _inputSize;
// 	this->outputSize = _outputSize;

// 	this->processingLayer = make_shared<DenseLayer>(this->inputSize, this->outputSize);
// 	if (messaging) {
// 		this->messagingLayer = make_shared<DenseLayer>(this->outputSize * 2, this->outputSize);
// 	}
// 	if (attention) {
// 		this->attentionLayer = make_shared<DenseLayer>(this->outputSize, 1);
// 	}
// }

// void GraphLayer::initialize(NeurologicalComponentInitializationInfo info)
// {
// 	this->processingLayer->initialize(info);
// 	if (this->messagingLayer.has_value()) {
// 		this->messagingLayer.value()->initialize(info);
// 	}
// 	if (this->attentionLayer.has_value()) {
// 		NeurologicalComponentInitializationInfo attentionInfo = info;
// 		attentionInfo.clippingMethod = nullopt;
// 		this->attentionLayer.value()->initialize(attentionInfo);
// 	}
// }

// int GraphLayer::getOutputSize()
// {
// 	return this->outputSize;
// }

// NeurologicalPassingValues GraphLayer::forwardPass(NeurologicalPassingValues _input, const string& sampleIndex)
// {
// 	neurologicalGraphNodes input = get<neurologicalGraphNodes>(_input);
// 	neurologicalGraphNodes output;
// 	neurologicalGraphNodes aggregatedOutputs;

// 	unordered_map<string, neurologicalListValues> nodeProcesses;

// 	for (pair<string, neurologicalGraphNode> element : input) {
// 		string nodeName = element.first;
// 		neurologicalGraphNode node = element.second;

// 		neurologicalListValues nodeProcessing = get<neurologicalListValues>
// 			(this->processingLayer->forwardPass(node.values, sampleIndex + "*" + nodeName));

// 		nodeProcesses[nodeName] = nodeProcessing;

// 		output[nodeName] = { .values = nodeProcessing, .bridges = node.bridges };
// 		aggregatedOutputs[nodeName] = output[nodeName];
// 	}

// 	unordered_map<string, unordered_map<string, neurologicalValue>> sampleAlphas;
// 	unordered_map<string, unordered_map<string, neurologicalListValues>> sampleMessages;

// 	// AGGREGATION
// 	for (pair<string, neurologicalGraphNode> element : input) {
// 		string nodeName = element.first;
// 		neurologicalGraphNode node = element.second;

// 		if (this->selectedNode.has_value()) {
// 			if (this->selectedNode.value() != nodeName) {
// 				continue;
// 			}
// 		}

// 		unordered_map<string, neurologicalListValues> messages(input.size());
// 		if (this->attentionLayer.has_value()) {
// 			// ATTENTION AGGREGATION

// 			unordered_map<string, neurologicalValue> scores(input.size());
// 			neurologicalValue scoresTotal = 0.0;

// 			unordered_map<string, neurologicalValue> alphas(input.size());

// 			for (string connectedNodeName : node.bridges) {
// 				neurologicalGraphNode connectedNode = input[connectedNodeName];

// 				neurologicalListValues message;
// 				if (this->messagingLayer.has_value()) {
// 					message = get<neurologicalListValues>
// 						(this->messagingLayer.value()->forwardPass(
// 							joinLists<neurologicalValue>({ output[nodeName].values, output[connectedNodeName].values }),
// 							sampleIndex + "*" + nodeName + "->" + connectedNodeName));
// 				}
// 				else {
// 					message = output[connectedNodeName].values;
// 				}

// 				neurologicalValue score = get<neurologicalListValues>
// 					(this->attentionLayer.value()->forwardPass(
// 						joinLists<neurologicalValue>({ message }),
// 						sampleIndex + "*" + nodeName + "->" + connectedNodeName))[0];

// 				scores[connectedNodeName] = score;
// 				scoresTotal += exp(score);
// 			}
// 			for (string connectedNodeName : node.bridges) {
// 				neurologicalListValues message = messages[connectedNodeName];

// 				alphas[connectedNodeName] = exp(scores[connectedNodeName]) / scoresTotal;

// 				messages[connectedNodeName] = message;

// 				for (int i = 0; i < message.size(); i++) {
// 					message[i] *= alphas[connectedNodeName];
// 				}

// 				aggregatedOutputs[nodeName].values = addListsValues<neurologicalValue>({ aggregatedOutputs[nodeName].values, message });
// 			}
// 			sampleAlphas[nodeName] = alphas;
// 		}
// 		else {
// 			// MEAN AGGREGATION
// 			for (string connectedNodeName : node.bridges) {
// 				neurologicalGraphNode connectedNode = input[connectedNodeName];

// 				neurologicalListValues message;
// 				if (this->messagingLayer.has_value()) {
// 					message = get<neurologicalListValues>
// 						(this->messagingLayer.value()->forwardPass(
// 							joinLists<neurologicalValue>({ output[nodeName].values, output[connectedNodeName].values }),
// 							sampleIndex + "*" + nodeName + "->" + connectedNodeName));
// 				}
// 				else {
// 					message = output[connectedNodeName].values;
// 				}

// 				messages[connectedNodeName] = message;

// 				// default aggregation [ divide by size ]
// 				for (int i = 0; i < message.size(); i++) {
// 					message[i] /= node.bridges.size();
// 				}

// 				aggregatedOutputs[nodeName].values = addListsValues<neurologicalValue>({ aggregatedOutputs[nodeName].values, message });
// 			}
// 		}

// 		sampleMessages[nodeName] = messages;
// 	}

// 	{
// 		lock_guard<mutex> lock(this->output_mutex);
// 		this->recordedAlphas[sampleIndex] = sampleAlphas;
// 		this->recordedMessages[sampleIndex] = sampleMessages;
// 		this->recordedOutputs[sampleIndex] = aggregatedOutputs;
// 	}

// 	if (this->selectedNode.has_value()) {
// 		return aggregatedOutputs[this->selectedNode.value()].values;
// 	}
// 	return aggregatedOutputs;
// }

// NeurologicalPassingValues GraphLayer::backPropagation(NeurologicalPassingValues _propagation, const string& sampleIndex)
// {
// 	neurologicalGraphNodes propagation;
// 	if (this->selectedNode.has_value()) {
// 		propagation[this->selectedNode.value()] = {.values = get<neurologicalListValues>(_propagation), .bridges = {}};
// 	}
// 	else {
// 		propagation = get<neurologicalGraphNodes>(_propagation);
// 	}
// 	neurologicalGraphNodes gradient;

// 	unordered_map<string, neurologicalListValues> nodesProcessingsPropagation;

// 	unordered_map<string, unordered_map<string, unordered_map<string, neurologicalValue>>> _recordedAlphas;
// 	unordered_map<string, unordered_map<string, unordered_map<string, neurologicalListValues>>> _recordedMessages;
// 	unordered_map<string, neurologicalGraphNodes> _recordedOutputs;
// 	{
// 		lock_guard<mutex> lock(this->output_mutex);
// 		_recordedAlphas = this->recordedAlphas;
// 		_recordedMessages = this->recordedMessages;
// 		_recordedOutputs = this->recordedOutputs;
// 	}

// 	// calculated propagation
// 	for (pair<string, neurologicalGraphNode> element : propagation) {
// 		string nodeName = element.first;
// 		neurologicalGraphNode outputNode = _recordedOutputs[sampleIndex][nodeName];
// 		neurologicalListValues nodePropagationValues = element.second.values;

// 		// initialize node propagation
// 		if (nodesProcessingsPropagation.count(nodeName)) {
// 			nodesProcessingsPropagation[nodeName] = addListsValues<neurologicalValue>
// 				({ nodesProcessingsPropagation[nodeName], nodePropagationValues });
// 		}
// 		else {
// 			nodesProcessingsPropagation[nodeName] = nodePropagationValues;
// 		}

// 		// initialize attention variables (if required)
// 		unordered_map<string, neurologicalValue> nodeAlphas = _recordedAlphas[sampleIndex][nodeName];
// 		unordered_map<string, neurologicalListValues> nodeMessages = _recordedMessages[sampleIndex][nodeName];

// 		unordered_map<string, neurologicalValue> scoreGradient;
// 		if (this->attentionLayer.has_value()) {
// 			for (pair<string, neurologicalListValues> messageElement : nodeMessages) {
// 				scoreGradient[messageElement.first] = 0.0;
// 				for (int j = 0; j < messageElement.second.size(); j++) {
// 					//cout << nodePropagationValues.size();
// 					//cout << endl;
// 					//cout << outputNode.bridges.size();
// 					//cout << endl;
// 					//cout << _propagation.index();
// 					//cout << " indexx?";
// 					//cout << endl;
// 					scoreGradient[messageElement.first] += messageElement.second[j] * nodePropagationValues[j];
// 				}
// 			}
// 		}

// 		// calculate connectedNodes propagations
// 		for (string connectedNodeName : outputNode.bridges) {
// 			neurologicalListValues bridgePropagation;

// 			// calculate bridgeProgation (attention)
// 			if (this->attentionLayer.has_value()) {
// 				neurologicalValue alpha = nodeAlphas[connectedNodeName];
// 				bridgePropagation = nodePropagationValues;

// 				neurologicalValue scoreAdjustment = 0.0;

// 				for (pair<string, neurologicalValue> alphaElement : nodeAlphas) {
// 					if (alphaElement.first == connectedNodeName) {
// 						scoreAdjustment += scoreGradient[alphaElement.first] * (alphaElement.second * (1 - alphaElement.second));
// 					}
// 					else {
// 						scoreAdjustment += scoreGradient[alphaElement.first] * (-alphaElement.second*alpha);
// 					}
// 				}
// 				neurologicalListValues attentionGradient = get<neurologicalListValues>
// 					(this->attentionLayer.value()->backPropagation(
// 						neurologicalListValues{ scoreAdjustment },
// 						sampleIndex + "*" + nodeName + "->" + connectedNodeName));

// 				//neurologicalListValues mainNodeAttentionPropagation = cutFromList<neurologicalValue>(attentionGradient, 1, this->outputSize);
// 				//neurologicalListValues bridgeNodeAttentionPropagation = cutFromList<neurologicalValue>(attentionGradient, 1, this->outputSize);

// 				//nodesProcessingsPropagation[nodeName] = addListsValues<neurologicalValue>
// 				//	({ nodesProcessingsPropagation[nodeName], mainNodeAttentionPropagation });

// 				for (int i = 0; i < bridgePropagation.size(); i++) {
// 					bridgePropagation[i] *= alpha;
// 					bridgePropagation[i] += attentionGradient[i];
// 				}

// 				// calculate messagePropagation (if required)
// 				if (this->messagingLayer.has_value()) {
// 					neurologicalListValues messageGradient = get<neurologicalListValues>
// 						(this->messagingLayer.value()->backPropagation(
// 							bridgePropagation,
// 							sampleIndex + "*" + nodeName + "->" + connectedNodeName));

// 					neurologicalListValues mainNodeMessagePropagation = cutFromList<neurologicalValue>(messageGradient, 1, this->outputSize);
// 					neurologicalListValues connectedNodeMessagePropagation = cutFromList<neurologicalValue>(messageGradient, this->outputSize+1, this->outputSize*2);

// 					nodesProcessingsPropagation[nodeName] = addListsValues<neurologicalValue>
// 						({ nodesProcessingsPropagation[nodeName], mainNodeMessagePropagation });

// 					if (nodesProcessingsPropagation.count(connectedNodeName)) {
// 						nodesProcessingsPropagation[connectedNodeName] = addListsValues<neurologicalValue>
// 							({ nodesProcessingsPropagation[connectedNodeName], connectedNodeMessagePropagation });
// 					}
// 					else {
// 						nodesProcessingsPropagation[connectedNodeName] = connectedNodeMessagePropagation;
// 					}
// 				}
// 				else {
// 					if (nodesProcessingsPropagation.count(connectedNodeName)) {
// 						nodesProcessingsPropagation[connectedNodeName] = addListsValues<neurologicalValue>
// 							({ nodesProcessingsPropagation[connectedNodeName], bridgePropagation });
// 					}
// 					else {
// 						nodesProcessingsPropagation[connectedNodeName] = bridgePropagation;
// 					}
// 				}
// 			}
// 			// calculate bridgeProgation (divide by size)
// 			else {
// 				bridgePropagation = nodePropagationValues;

// 				for (int i = 0; i < bridgePropagation.size(); i++) {
// 					bridgePropagation[i] /= outputNode.bridges.size();
// 				}

// 				// calculate messagePropagation (if required)
// 				if (this->messagingLayer.has_value()) {
// 					neurologicalListValues messageGradient = get<neurologicalListValues>
// 						(this->messagingLayer.value()->backPropagation(
// 							bridgePropagation,
// 							sampleIndex + "*" + nodeName + "->" + connectedNodeName));

// 					neurologicalListValues mainNodeMessagePropagation = cutFromList<neurologicalValue>(messageGradient, 1, this->outputSize);
// 					neurologicalListValues connectedNodeMessagePropagation = cutFromList<neurologicalValue>(messageGradient, this->outputSize+1, this->outputSize * 2);

// 					nodesProcessingsPropagation[nodeName] = addListsValues<neurologicalValue>
// 						({ nodesProcessingsPropagation[nodeName], mainNodeMessagePropagation });

// 					if (nodesProcessingsPropagation.count(connectedNodeName)) {
// 						nodesProcessingsPropagation[connectedNodeName] = addListsValues<neurologicalValue>
// 							({ nodesProcessingsPropagation[connectedNodeName], connectedNodeMessagePropagation });
// 					}
// 					else {
// 						nodesProcessingsPropagation[connectedNodeName] = connectedNodeMessagePropagation;
// 					}
// 				}
// 				else {
// 					if (nodesProcessingsPropagation.count(connectedNodeName)) {
// 						nodesProcessingsPropagation[connectedNodeName] = addListsValues<neurologicalValue>
// 							({ nodesProcessingsPropagation[connectedNodeName], bridgePropagation });
// 					}
// 					else {
// 						nodesProcessingsPropagation[connectedNodeName] = bridgePropagation;
// 					}
// 				}
// 			}
// 		}
// 	}

// 	for (pair<string, neurologicalListValues> element : nodesProcessingsPropagation) {
// 		string nodeName = element.first;
// 		neurologicalListValues nodePropagation = element.second;

// 		gradient[nodeName] = { .values = get<neurologicalListValues>(
// 			this->processingLayer->backPropagation(nodePropagation, sampleIndex + "*" + nodeName))
// 		, .bridges = {} };
// 	}

// 	return gradient;
// }

// void GraphLayer::applyDerivatives(derivativesApplyingInfo applyingInfo) {
// 	this->processingLayer->applyDerivatives(applyingInfo);
// 	if (this->messagingLayer.has_value()) {
// 		this->messagingLayer.value()->applyDerivatives(applyingInfo);
// 	}
// 	if (this->attentionLayer.has_value()) {
// 		this->attentionLayer.value()->applyDerivatives(applyingInfo);
// 	}

// 	this->recordedAlphas.clear();
// 	this->recordedMessages.clear();
// 	this->recordedOutputs.clear();
// }

// shared_ptr<GraphLayer> GraphLayer::selectNode(string nodeName)
// {
// 	this->selectedNode = nodeName;
// 	return this->shared_from_this();
// }