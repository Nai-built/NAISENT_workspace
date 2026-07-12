#pragma once

#include "../INeurologicalComponent.h"
#include "../IChannelsPooler.h"

struct Max : public IChannelsPooler
{
public:
    vector<int> picking_mask;

    Max(int _poolingWidth, int _poolingHeight, int _stride);
};