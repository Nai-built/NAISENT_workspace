#include "ReLU.h"

ReLU::ReLU(neurologicalValue _leakage) {
    this->activationType = ActivationFunctionTypes::RELU;

    this->leakage = _leakage;
}