// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include <vector>
#include <initializer_list>
#include <random>

#include <stdio.h>
#include <stdlib.h>

// using namespace std;

/**
 * 
 */

std::vector<std::string> splitString(const std::string& str, char delimiter);
std::vector<std::string> splitStringByString(const std::string& s, const std::string& delimiter);

static std::random_device rd;
static std::default_random_engine r_engine(rd());

template<typename T>
T getRandom(T min, T max);

template<typename T>
T getRandom(T min, T max, std::default_random_engine& _r_engine);

template<typename T>
std::vector<T> joinLists(std::vector<std::vector<T>> lists);

template<typename T>
std::vector<T> addListsValues(std::vector<std::vector<T>> lists);

template<typename T>
std::vector<T> cutFromList(std::vector<T> lists, int start, int end);

template<typename T>
bool checkInBounds(const std::vector<T>& list, const int& index);

template<typename T>
inline void zeroOutList(std::vector<T>& buffer);

#include "Extra.inl"