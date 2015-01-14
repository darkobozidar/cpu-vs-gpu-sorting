#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

#include "../Utils/data_types_common.h"


void allocHostMemory(
    data_t **input, data_t **outputParallel, data_t **outputSequential, data_t **outputCorrect,
    uint_t **globalBucketOffsets, double ***timers, uint_t tableLen, uint_t testRepetitions
);
void freeHostMemory(
    data_t *input, data_t *outputParallel, data_t *outputSequential, data_t *outputCorrect,
    uint_t *globalBucketOffsets, double **timers
);

void allocDeviceMemory(
    data_t **dataTable, data_t **dataBuffer, data_t **samplesLocal, data_t **samplesGlobal,
    uint_t **localBucketSizes, uint_t **localBucketOffsets, uint_t **globalBucketOffsets, uint_t tableLen
);
void freeDeviceMemory(
    data_t *dataTable, data_t *dataBuffer, data_t *samplesLocal, data_t *samplesGlobal, uint_t *localBucketSizes,
    uint_t *localBucketOffsets, uint_t *globalBucketOffsets
);

#endif
