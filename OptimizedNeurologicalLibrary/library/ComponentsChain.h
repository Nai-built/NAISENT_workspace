#pragma once

#include "INeurologicalComponent.h"

#include <functional>

// typedef std::vector<std::function<void(const neurologicalConstantSpan input, const neurologicalSpan& output)>> chainBuild;

struct ComponentsChain : public INeurologicalComponent
{
    std::vector<neurologicalBuffer> chainBuffers;
    std::vector<neurologicalBuffer> chainPropagations;

    std::vector<NeurologicalComponent_UNIQUE> componentsChain;
    
    ComponentsChain();
    
    static void Initialize(ComponentsChain* chain);
    static void ActivateChain(ComponentsChain* chain, const neurologicalConstantSpan input, const neurologicalSpan output, const int& batchSize
    , const lengths seriesLengths, const int& totalSamples);
    static void AdjustChain(ComponentsChain* chain, const neurologicalSpan propagation, const neurologicalSpan inputPropagation, const neurologicalValue& learnRate, const int& batchSize
    , const lengths seriesLengths, const int& totalSamples);
};