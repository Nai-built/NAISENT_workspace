#include "Min.h"

Min::Min(int _poolingWidth, int _poolingHeight, int _stride) {
    this->poolerType = ChannelsPoolerTypes::MIN;

    this->poolingWidth = _poolingWidth;
    this->poolingHeight = _poolingHeight;
    this->stride = _stride;
}