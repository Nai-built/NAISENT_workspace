#pragma once

#include "../INeurologicalComponent.h"
#include "../IChannelsPooler.h"

struct Min : public IChannelsPooler
{
public:
    vector<int> picking_mask;

    Min(int _poolingWidth, int _poolingHeight, int _stride);
};