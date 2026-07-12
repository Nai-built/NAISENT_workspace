#include "Max.h"

Max::Max(int _poolingWidth, int _poolingHeight, int _stride) {
    this->poolerType = ChannelsPoolerTypes::MAX;

    this->poolingWidth = _poolingWidth;
    this->poolingHeight = _poolingHeight;
    this->stride = _stride;
}