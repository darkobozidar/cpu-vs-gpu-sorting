#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

#include "../Utils/data_types_common.h"


void allocHostMemory(
    data_t **inputKeys, data_t **inputValues, data_t **outputParallelKeys, data_t **outputParallelValues,
    data_t **inputSequentialKeys, data_t **inputSequentialValues, data_t **outputSequentialKeys,
    data_t **outputSequentialValues, data_t **outputCorrect, uint_t **countersSequential, double ***timers,
    uint_t tableLen, uint_t testRepetitions
);
void freeHostMemory(
    data_t *inputKeys, data_t *inputValues, data_t *outputParallelKeys, data_t *outputParallelValues,
    data_t *inputSequentialKeys, data_t *inputSequentialValues, data_t *outputSequentialKeys,
    data_t *outputSequentialValues, data_t *outputCorrect, uint_t *countersSequential, double **timers
);

void allocDeviceMemory(
    data_t **dataTableKeys, data_t **dataTableValues, data_t **dataBufferKeys, data_t **dataBufferValues,
    uint_t **bucketOffsetsLocal, uint_t **bucketOffsetsGlobal, uint_t **bucketSizes, uint_t tableLen
);
void freeDeviceMemory(
    data_t *dataTableKeys, data_t *dataTableValues, data_t *dataBufferKeys, data_t *dataBufferValues,
    uint_t *bucketOffsetsLocal, uint_t *bucketOffsetsGlobal, uint_t *bucketSizes
);

#endif
