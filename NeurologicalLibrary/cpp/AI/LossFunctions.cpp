// Fill out your copyright notice in the Description page of Project Settings.

#include "Neurological.h"
#include "LossFunctions.h"

neurologicalListValues zero_loss_gradient(int size)
{
	neurologicalListValues gradient(size);

	for (int i = 0; i < size; i++) {
		gradient[i] = 0;
	}

	return gradient;
}

neurologicalListValues MSE_loss_gradient(neurologicalListValues prediction, neurologicalListValues correction)
{
	int size = prediction.size();

	neurologicalListValues gradient(size);

	for (int i = 0; i < size; i++) {
		gradient[i] = prediction[i] - correction[i];
	}

	return gradient;
}

neurologicalValue MSE_loss_value(neurologicalListValues prediction, neurologicalListValues correction)
{
	int size = prediction.size();

	neurologicalValue value = 0;

	for (int i = 0; i < size; i++) {
		value += pow(prediction[i] - correction[i], 2);
	}

	return value / size;
}

neurologicalListValues KL_divergence_gradient(neurologicalListValues prediction, neurologicalListValues correction)
{
    
	int size = prediction.size();

	neurologicalListValues gradient(size);

	for (int i = 0; i < size; i++) {
        if (correction[i] > 0) {
		    gradient[i] = -correction[i] / prediction[i];
        } else {
            gradient[i] = 0;
        }
	}

	return gradient;
}

neurologicalValue KL_divergence_value(neurologicalListValues prediction, neurologicalListValues correction)
{
	int size = prediction.size();

	neurologicalValue value = 0;

	for (int i = 0; i < size; i++) {
        if (correction[i] > 0) {
            value += correction[i] * log(correction[i] / prediction[i]);
        }
	}

	return value / size;
}
