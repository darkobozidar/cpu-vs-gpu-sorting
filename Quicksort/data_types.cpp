#include "data_types.h"

/*
Because of circular dependencies between stuctures, methods have to be implemented after structure definitons.
*/

/* HostGlobalSequence */

void HostGlobalSequence::setInitSequence(uint_t tableLen, data_t initPivot) {
    start = 0;
    length = tableLen;
    oldStart = start;
    oldLength = length;
    pivot = initPivot;
    direction = false;
}

void HostGlobalSequence::setLowerSequence(h_glob_seq_t globalSeqHost, d_glob_seq_t globalSeqDev) {
    start = globalSeqHost.oldStart;
    length = globalSeqDev.offsetLower;
    oldStart = start;
    oldLength = length;
    pivot = (globalSeqDev.minVal + globalSeqHost.pivot) / 2;
    direction = !globalSeqHost.direction;
}

void HostGlobalSequence::setGreaterSequence(h_glob_seq_t globalSeqHost, d_glob_seq_t globalSeqDev) {
    start = globalSeqHost.oldStart + globalSeqHost.length - globalSeqDev.offsetGreater;
    length = globalSeqDev.offsetGreater;
    oldStart = start;
    oldLength = length;
    pivot = (globalSeqHost.pivot + globalSeqDev.maxVal) / 2;
    direction = !globalSeqHost.direction;
}


/* DeviceGlobalSequence */

void DeviceGlobalSequence::setSequence(h_glob_seq_t globalSeqHost, uint_t threadBlocksPerSequence) {
    start = globalSeqHost.start;
    length = globalSeqHost.length;
    direction = globalSeqHost.direction;
    pivot = globalSeqHost.pivot;
    blockCounter = threadBlocksPerSequence;

    offsetLower = 0;
    offsetGreater = 0;

    minVal = UINT32_MAX;
    maxVal = 0;
}
