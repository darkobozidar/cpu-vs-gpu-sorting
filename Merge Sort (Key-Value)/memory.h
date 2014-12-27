#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

#include "../Utils/data_types_common.h"


void allocHostMemory(
    data_t **inputKeys, data_t **inputValues, data_t **outputParallelKeys, data_t **outputParallelValues,
    data_t **outputSequentialKeys, data_t **outputSequentialValues, data_t **bufferSequentialKeys,
    data_t **bufferSequentialValues, data_t **outputCorrect, double ***timers, uint_t tableLen, uint_t testRepetitions
);
void freeHostMemory(
    data_t *inputKeys, data_t *inputValues, data_t *outputParallelKeys, data_t *outputParallelValues,
    data_t *outputSequentialKeys, data_t *outputSequentialValues, data_t *bufferSequentialKeys,
    data_t *bufferSequentialValues, data_t *outputCorrect, double **timers
);

void allocDeviceMemory(
    data_t **dataKeys, data_t **dataValues, data_t **bufferKeys, data_t **bufferValues, uint_t **ranksEven,
    uint_t **ranksOdd, uint_t tableLen
);
void freeDeviceMemory(
    data_t *dataKeys, data_t *dataValues, data_t *bufferKeys, data_t *bufferValues, uint_t *ranksEven,
    uint_t *ranksOdd
);

#endif
