#include "LossFunctions.h"

void MSE_loss_propagation(const neurologicalConstantSpan prediction, const neurologicalConstantSpan correction, const neurologicalSpan propagation) {
    for (int i = 0; i < prediction.size(); ++i) {
        propagation[i] = prediction[i] - correction[i];
    }
}