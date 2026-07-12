// Fill out your copyright notice in the Description page of Project Settings.

#include "../Extra.h"

#include <vector>

using namespace std;

template<typename T>
vector<T> CustomVector2::toList()
{
	vector<T> list(2);
	list[0] = static_cast<T>(this->x);
	list[1] = static_cast<T>(this->y);

	return list;
}

template<typename T>
CustomVector2 CustomVector2::FromList(vector<T> list)
{
	return CustomVector2(list[0], list[1]);
}