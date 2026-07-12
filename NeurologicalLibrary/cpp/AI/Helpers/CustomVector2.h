// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "../Extra.h"

#include <vector>
#include <string>

using namespace std;

/**
 * 
 */
struct CustomVector2
{
	float x;
	float y;

	CustomVector2();
	CustomVector2(float _x, float _y);

	string toString();

	float magnitude();
	CustomVector2 unit();

	CustomVector2 negate();
	CustomVector2 round();

	CustomVector2 negate(CustomVector2 vector2);
	CustomVector2 add(CustomVector2 vector2);
	CustomVector2 multiply(float multiplication);

	bool isIn(float bounds);

	template<typename T = float>
	vector<T> toList();

	static CustomVector2 RandomVector2(const float range);

	template<typename T = float>
	static CustomVector2 FromList(vector<T> list);
};

#include "CustomVector2.inl"