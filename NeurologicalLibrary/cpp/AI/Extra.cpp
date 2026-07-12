// Fill out your copyright notice in the Description page of Project Settings.


#include "Extra.h"
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

// GEMINI
std::vector<std::string> splitString(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    // Read from the stringstream into 'token' until the delimiter is found
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    
    return tokens;
}
std::vector<std::string> splitStringByString(const std::string& s, const std::string& delimiter) {
    std::vector<std::string> tokens;
    size_t last = 0;
    size_t next = 0;
    while ((next = s.find(delimiter, last)) != std::string::npos) {
        tokens.push_back(s.substr(last, next - last));
        last = next + delimiter.length();
    }
    // Add the last token (part after the last delimiter, or the whole string if no delimiter found)
    tokens.push_back(s.substr(last));
    return tokens;
}

// OTHER DEFEINITIONS ARE INSIDE Extra.inl