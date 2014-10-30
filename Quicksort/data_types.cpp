#include "data_types.h"


/*
Because of circular dependencies between stuctures, methods have to be implemented after structure definitons.
*/

/* HostGlobalSequence */

void HostGlobalSequence::setInitSeq(uint_t tableLen, data_t initPivot) {
    start = 0;
    length = tableLen;
    oldStart = start;
    oldLength = length;
    pivot = initPivot;
    direction = PRIMARY_MEM_TO_BUFFER;
}

void HostGlobalSequence::setLowerSeq(h_glob_seq_t globalSeqHost, d_glob_seq_t globalSeqDev) {
    start = globalSeqHost.oldStart;
    length = globalSeqDev.offsetLower;
    oldStart = start;
    oldLength = length;
    pivot = (globalSeqDev.minVal + globalSeqHost.pivot) / 2;
    direction = (TransferDirection) !globalSeqHost.direction;
}

void HostGlobalSequence::setGreaterSeq(h_glob_seq_t globalSeqHost, d_glob_seq_t globalSeqDev) {
    start = globalSeqHost.oldStart + globalSeqHost.length - globalSeqDev.offsetGreater;
    length = globalSeqDev.offsetGreater;
    oldStart = start;
    oldLength = length;
    pivot = (globalSeqHost.pivot + globalSeqDev.maxVal) / 2;
    direction = (TransferDirection) !globalSeqHost.direction;
}


/* DeviceGlobalSequence */

void DeviceGlobalSequence::setFromHostSeq(h_glob_seq_t globalSeqHost, uint_t startThreadBlock,
                                         uint_t threadBlocksPerSequence) {
    start = globalSeqHost.start;
    length = globalSeqHost.length;
    pivot = globalSeqHost.pivot;
    direction = globalSeqHost.direction;

    startThreadBlockIdx = startThreadBlock;
    threadBlockCounter = threadBlocksPerSequence;

    offsetLower = 0;
    offsetGreater = 0;

    minVal = UINT32_MAX;
    maxVal = 0;
}


/* LocalSequence */

void LocalSequence::setLowerSeq(h_glob_seq_t globalSeqHost, d_glob_seq_t globalSeqDev) {
    start = globalSeqHost.oldStart;
    length = globalSeqDev.offsetLower;
    direction = (TransferDirection) !globalSeqHost.direction;
}

void LocalSequence::setGreaterSeq(h_glob_seq_t globalSeqHost, d_glob_seq_t globalSeqDev) {
    start = globalSeqHost.oldStart + globalSeqHost.length - globalSeqDev.offsetGreater;
    length = globalSeqDev.offsetGreater;
    direction = (TransferDirection) !globalSeqHost.direction;
}

void LocalSequence::setFromGlobalSeq(h_glob_seq_t globalParams) {
    start = globalParams.start;
    length = globalParams.length;
    direction = globalParams.direction;
}
