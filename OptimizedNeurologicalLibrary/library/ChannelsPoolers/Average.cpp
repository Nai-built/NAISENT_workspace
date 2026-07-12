#include "Average.h"

Average::Average(int _poolingWidth, int _poolingHeight, int _stride) {
    this->poolerType = ChannelsPoolerTypes::AVERAGE;

    this->poolingWidth = _poolingWidth;
    this->poolingHeight = _poolingHeight;
    this->stride = _stride;
}