#pragma once

#include "../INeurologicalComponent.h"
#include "../IChannelsPooler.h"

struct Average : public IChannelsPooler
{
public:
    Average(int _poolingWidth, int _poolingHeight, int _stride);
};