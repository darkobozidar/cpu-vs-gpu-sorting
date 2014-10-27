#include "data_types.h"

/*
Because of circular dependencies between stuctures, methods have to be implemented after structure definitons.
*/

void HostGlobalSequence::setDefaultParams(uint_t tableLen) {
    start = 0;
    length = tableLen;
    oldStart = start;
    oldLength = length;
    direction = false;
}

void HostGlobalSequence::lowerSequence(h_glob_seq_t oldParams, d_gparam_t deviceParams) {
    start = oldParams.oldStart;
    length = deviceParams.offsetLower;
    oldStart = start;
    oldLength = length;

    direction = !oldParams.direction;
    pivot = (deviceParams.minVal + oldParams.pivot) / 2;
}

void HostGlobalSequence::greaterSequence(h_glob_seq_t oldParams, d_gparam_t deviceParams) {
    start = oldParams.oldStart + oldParams.length - deviceParams.offsetGreater;
    length = deviceParams.offsetGreater;
    oldStart = start;
    oldLength = length;

    direction = !oldParams.direction;
    pivot = (oldParams.pivot + deviceParams.maxVal) / 2;
}
