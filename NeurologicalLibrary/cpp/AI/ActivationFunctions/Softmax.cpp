// Fill out your copyright notice in the Description page of Project Settings.


#include "Softmax.h"

Softmax::Softmax()
{
    this->ignoreJacobian = false;
}
Softmax::Softmax(bool ignoreJacobian)
{
    this->ignoreJacobian = ignoreJacobian;
}

void Softmax::__activation(const neurologicalListValues& _listInput, neurologicalListValues& _listOutput)
{
    // neurologicalValue largestNumber = -HUGE;
	// for (int i = 0; i < _listInput.size(); i++) {
    //     if (_listInput[i] > largestNumber) {
    //         largestNumber = _listInput[i];
    //     }
	// }
    neurologicalValue expSum = 0;

	for (int i = 0; i < _listInput.size(); i++) {
        expSum += exp(_listInput[i]);
	}
	for (int i = 0; i < _listInput.size(); i++) {
        _listOutput[i] = exp(_listInput[i])/expSum;
	}
}

void Softmax::_jacobian(const neurologicalListValues& _softmax, vector<neurologicalListValues>& _jacobianResult) {
    for (int i = 0; i < _softmax.size(); i++) {   
        _jacobianResult.push_back(neurologicalListValues(_softmax.size()));
        for (int j = 0; j < _softmax.size(); j++) {
            if (i == j) {
                _jacobianResult[i][j] += _softmax[i] * (((double)1.0) - _softmax[i]);
            } else {
                _jacobianResult[i][j] += (-_softmax[i]) * _softmax[j];
            }
        }
    }
}

void Softmax::__gradient(bool ignoreJacobian, const neurologicalListValues& _listPropagation, const neurologicalListValues& _listInput, const neurologicalListValues& _listOutput, neurologicalListValues& _listInputPropagation)
{
    if (!ignoreJacobian) {
        vector<neurologicalListValues> jacobianResult(_listOutput.size());
        _jacobian(_listOutput, jacobianResult);

        for (int i = 0; i < jacobianResult.size(); i++) {
            for (int j = 0; j < _listPropagation.size(); j++) {
                _listInputPropagation[j] += jacobianResult[i][j] * _listPropagation[j];
                
                // cout << i << ", " << j << ": " << jacobianResult[i][j] << ", " << _listPropagation[j] << ", " << _gradient[j] << endl;
            }
        }
    }

    _listInputPropagation = _listPropagation;
}

void Softmax::activation(const neurologicalListValues& _listInput, neurologicalListValues& _listOutput)
{
	this->__activation(_listInput, _listOutput);
}

void Softmax::gradient(const neurologicalListValues& _listPropagation, const neurologicalListValues& _listInput, const neurologicalListValues& _listOutput, neurologicalListValues& _listInputPropagation)
{
	this->__gradient(this->ignoreJacobian, _listPropagation, _listInput, _listOutput, _listInputPropagation);
}