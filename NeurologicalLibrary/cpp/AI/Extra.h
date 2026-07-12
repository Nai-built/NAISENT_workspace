// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include <vector>
#include <initializer_list>
#include <random>

#include <stdio.h>
#include <stdlib.h>

using namespace std;

/**
 * 
 */

vector<string> splitString(const string& str, char delimiter);
vector<string> splitStringByString(const string& s, const string& delimiter);

static random_device rd;
static default_random_engine r_engine(rd());

template<typename T>
T getRandom(T min, T max);

template<typename T>
T getRandom(T min, T max, default_random_engine& _r_engine);

template<typename T>
vector<T> joinLists(vector<vector<T>> lists);

template<typename T>
vector<T> addListsValues(vector<vector<T>> lists);

template<typename T>
vector<T> cutFromList(vector<T> lists, int start, int end);

template<typename T>
bool checkInBounds(const vector<T>& list, const int& index);

#include "Extra.inl"