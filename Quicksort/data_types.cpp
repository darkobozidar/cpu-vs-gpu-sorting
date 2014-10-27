//#include "data_types.h"
//
//
//void HostGlobalParams::lowerSequence(h_gparam_t oldParams, d_gparam_t deviceParams) {
//    start = oldParams.oldStart;
//    length = deviceParams.offsetLower;
//    oldStart = start;
//    oldLength = length;
//
//    direction = !oldParams.direction;
//    pivot = (deviceParams.minVal + oldParams.pivot) / 2;
//}
//
//void HostGlobalParams::greaterSequence(h_gparam_t oldParams, d_gparam_t deviceParams) {
//    start = oldParams.oldStart + oldParams.length - deviceParams.offsetGreater;
//    length = deviceParams.offsetGreater;
//    oldStart = start;
//    oldLength = length;
//
//    direction = !oldParams.direction;
//    pivot = (oldParams.pivot + deviceParams.maxVal) / 2;
//}
