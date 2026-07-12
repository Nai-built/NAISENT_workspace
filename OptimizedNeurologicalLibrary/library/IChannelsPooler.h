#pragma once

#include <vector>
#include <span>
#include <memory>

enum class ChannelsPoolerTypes {
    NONE = 0,

    MAX = 1,
    AVERAGE = 2,
    MIN = 3,
};

struct IChannelsPooler
{
public:
    neurologicalBuffer cached_output;
    neurologicalBuffer processedPropagation;
    
    ChannelsPoolerTypes poolerType;
    
    int poolingWidth;
    int poolingHeight;

    int stride;

    virtual ~IChannelsPooler() = default; // Make the base polymorphic
};

typedef std::unique_ptr<IChannelsPooler> ChannelsPooler_UNIQUE;
typedef std::weak_ptr<IChannelsPooler> ChannelsPoolerPTR;