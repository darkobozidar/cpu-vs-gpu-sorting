#include "data_types.h"


/*
Because of circular dependencies between stuctures, structure methods have to be implemented after
structure definitions.
*/

/* HostGlobalSequence */

void HostGlobalSequence::setInitSeq(uint_t tableLen, data_t initMinVal, data_t initMaxVal)
{
    start = 0;
    length = tableLen;
    minVal = initMinVal;
    maxVal = initMaxVal;
    direction = PRIMARY_MEM_TO_BUFFER;
}

void HostGlobalSequence::setLowerSeq(h_glob_seq_t globalSeqHost, d_glob_seq_t globalSeqDev)
{
    start = globalSeqHost.start;
    length = globalSeqDev.offsetLower;
    minVal = globalSeqHost.minVal;
#if USE_REDUCTION_IN_GLOBAL_SORT
    maxVal = globalSeqDev.lowerSeqMaxVal;
#else
    maxVal = globalSeqDev.pivot;
#endif
    direction = (direct_t)!globalSeqHost.direction;
}

void HostGlobalSequence::setGreaterSeq(h_glob_seq_t globalSeqHost, d_glob_seq_t globalSeqDev)
{
    start = globalSeqHost.start + globalSeqHost.length - globalSeqDev.offsetGreater;
    length = globalSeqDev.offsetGreater;
#if USE_REDUCTION_IN_GLOBAL_SORT
    minVal = globalSeqDev.greaterSeqMinVal;
#else
    minVal = globalSeqDev.pivot;
#endif
    maxVal = globalSeqHost.maxVal;
    direction = (direct_t)!globalSeqHost.direction;
}


/* DeviceGlobalSequence */

void DeviceGlobalSequence::setFromHostSeq(
    h_glob_seq_t globalSeqHost, uint_t startThreadBlock, uint_t threadBlocksPerSequence
)
{
    start = globalSeqHost.start;
    length = globalSeqHost.length;
    direction = globalSeqHost.direction;

    // Calculates avg. of min and max value, even if sum is greater than max value of data type
    data_t sum = (globalSeqHost.minVal + globalSeqHost.maxVal);
    if (sum < globalSeqHost.maxVal)
    {
        pivot = (MAX_VAL / 2) + (sum / 2) + 1;
    }
    else
    {
        pivot = sum / 2;
    }

    startThreadBlockIdx = startThreadBlock;
    threadBlockCounter = threadBlocksPerSequence;

    offsetLower = 0;
    offsetGreater = 0;
    offsetPivotValues = 0;

#if USE_REDUCTION_IN_GLOBAL_SORT
    greaterSeqMinVal = MAX_VAL;
    lowerSeqMaxVal = MIN_VAL;
#endif
}


/* LocalSequence */

void LocalSequence::setInitSeq(uint_t tableLen)
{
    start = 0;
    length = tableLen;
    direction = PRIMARY_MEM_TO_BUFFER;
}

void LocalSequence::setLowerSeq(h_glob_seq_t globalSeqHost, d_glob_seq_t globalSeqDev)
{
    start = globalSeqHost.start;
    length = globalSeqDev.offsetLower;
    direction = (direct_t)!globalSeqHost.direction;
}

void LocalSequence::setGreaterSeq(h_glob_seq_t globalSeqHost, d_glob_seq_t globalSeqDev)
{
    start = globalSeqHost.start + globalSeqHost.length - globalSeqDev.offsetGreater;
    length = globalSeqDev.offsetGreater;
    direction = (direct_t)!globalSeqHost.direction;
}
