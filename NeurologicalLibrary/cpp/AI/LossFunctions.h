// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "Neurological.h"

neurologicalListValues zero_loss_gradient(int size);

neurologicalListValues MSE_loss_gradient(neurologicalListValues prediction, neurologicalListValues correction);
neurologicalValue MSE_loss_value(neurologicalListValues prediction, neurologicalListValues correction);

neurologicalListValues KL_divergence_gradient(neurologicalListValues prediction, neurologicalListValues correction);
neurologicalValue KL_divergence_value(neurologicalListValues prediction, neurologicalListValues correction);