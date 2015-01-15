#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

#include "../Utils/data_types_common.h"


void allocHostMemory(
    data_t **inputKeys, data_t **inputValues, data_t **outputParallelKeys, data_t **outputParallelValues,
    data_t **inputSequentialKeys, data_t **inputSequentialValues, data_t **bufferSequentialKeys,
    data_t **bufferSequentialValues, data_t **outputSequentialKeys, data_t **outputSequentialValues,
    data_t **outputCorrect, data_t **samples, uint_t **elementBuckets, uint_t **globalBucketOffsets,
    double ***timers, uint_t tableLen, uint_t testRepetitions
);
void freeHostMemory(
    data_t *inputKeys, data_t *inputValues, data_t *outputParallelKeys, data_t *outputParallelValues,
    data_t *inputSequentialKeys, data_t *inputSequentialValues, data_t *bufferSequentialKeys,
    data_t *bufferSequentialValues, data_t *outputSequentialKeys, data_t *outputSequentialValues,
    data_t *outputCorrect, data_t *samples, uint_t *elementBuckets, uint_t *globalBucketOffsets, double **timers
);

void allocDeviceMemory(
    data_t **dataTableKeys, data_t **dataTableValues, data_t **d_dataBufferKeys, data_t **d_dataBufferValues,
    data_t **samplesLocal, data_t **samplesGlobal, uint_t **localBucketSizes, uint_t **localBucketOffsets,
    uint_t **globalBucketOffsets, uint_t tableLen
);
void freeDeviceMemory(
    data_t *dataTableKeys, data_t *dataTableValues, data_t *d_dataBufferKeys, data_t *d_dataBufferValues,
    data_t *samplesLocal, data_t *samplesGlobal, uint_t *localBucketSizes, uint_t *localBucketOffsets,
    uint_t *globalBucketOffsets
);

#endif
