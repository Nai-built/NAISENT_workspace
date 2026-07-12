// Fill out your copyright notice in the Description page of Project Settings.


#include <vector>
#include <random>

#include <stdio.h>
#include <stdlib.h>

using namespace std;

template<typename T>
T getRandom(T min, T max) {
	std::uniform_real_distribution<T> distribution(min, max);

	/*float _random = ((float)rand()) / RAND_MAX;

	float _range = (max - min) * _random;*/

	//return static_cast<T>(min + _range);

	return static_cast<T>(distribution(r_engine));
}

template<typename T>
T getRandom(T min, T max, std::default_random_engine& _r_engine) {
	std::uniform_real_distribution<T> distribution(min, max);

	/*float _random = ((float)rand()) / RAND_MAX;

	float _range = (max - min) * _random;*/

	//return static_cast<T>(min + _range);

	return static_cast<T>(distribution(_r_engine));
}

template<typename T>
std::vector<T> joinLists(std::vector<std::vector<T>> lists) {
	std::vector<T> newList;
	
	for (std::vector<T> _list : lists) {
		for (T _e : _list) {
			newList.push_back(_e);
		}
	}

	return newList;
}

template<typename T>
std::vector<T> addListsValues(std::vector<std::vector<T>> lists) {
	std::vector<T> newList;
	
	for (int i = 0; i < lists.size(); i++) {
		for (int j = 0; j < lists[i].size(); j++) {
			if (i == 0) {
				newList.push_back(lists[i][j]);
			}
			else {
				newList[j] += lists[i][j];
			}
		}
	}

	return newList;
}

template<typename T>
std::vector<T> cutFromList(std::vector<T> list, int start, int end)
{
	std::vector<T> newList;

	for (int i = start - 1; i < end; i++)
	{
		newList.push_back(list[i]);
	}

	return newList;
}

template<typename T>
bool checkInBounds(const std::vector<T>& list, const int& index)
{
	return index >= 0 && static_cast<size_t>(index) < list.size();
}

template<typename T>
inline void zeroOutList(std::vector<T>& list) {
    std::memset(list.data(), 0, list.size() * sizeof(T));
}
//