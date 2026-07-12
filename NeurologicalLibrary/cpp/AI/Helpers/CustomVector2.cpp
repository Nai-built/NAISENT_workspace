// Fill out your copyright notice in the Description page of Project Settings.


#include "CustomVector2.h"

#include "../Extra.h"

#include <iostream>
#include <vector>
#include <string>
#include <memory>

using namespace std;

CustomVector2::CustomVector2()
{
	this->x = 0;
	this->y = 0;
}
CustomVector2::CustomVector2(float _x, float _y)
{
	this->x = isnan(_x) ? 0 : _x;
	this->y = isnan(_y) ? 0 : _y;
}

string CustomVector2::toString()
{
	return "X: " + to_string(this->x) + " Y: " + to_string(this->y);
}

float CustomVector2::magnitude()
{
	return sqrt(pow(this->x, 2) + pow(this->y, 2));
}
CustomVector2 CustomVector2::unit()
{
	float _magnitude = this->magnitude();
	return CustomVector2(this->x/_magnitude, this->y/_magnitude);
}

CustomVector2 CustomVector2::negate()
{
	return CustomVector2(-this->x, -this->y);
}
CustomVector2 CustomVector2::round()
{
	return CustomVector2(std::round(this->x), std::round(this->y));
}

CustomVector2 CustomVector2::negate(CustomVector2 vector2)
{
	return CustomVector2(this->x - vector2.x, this->y - vector2.y);
}
CustomVector2 CustomVector2::add(CustomVector2 vector2)
{
	return CustomVector2(this->x + vector2.x, this->y + vector2.y);
}
CustomVector2 CustomVector2::multiply(float multiplication)
{
	return CustomVector2(this->x * multiplication, this->y * multiplication);
}
bool CustomVector2::isIn(float bounds)
{
	return (this->x <= bounds && this->x >= -bounds) && (this->y <= bounds && this->y >= -bounds);
}

// "toList" DEFINITION IS INSIDE CustomVector2.inl

CustomVector2 CustomVector2::RandomVector2(const float range)
{
	//T t = getRandom<T>(-.1, .1);
	return CustomVector2(getRandom<float>(-range, range), getRandom<float>(-range, range));
}

// "FromList" DEFINITION IS INSIDE CustomVector2.inl