#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

#include "../Utils/data_types_common.h"
#include "data_types.h"


void allocHostMemory(
    data_t **input, data_t **outputParallel, data_t **outputSequential, data_t **outputCorrect,
    double ***timers, uint_t tableLen, uint_t testRepetitions
);
void freeHostMemory(
    data_t *input, data_t *outputParallel, data_t *outputSequential, data_t *outputCorrect,
    double **timers
);

void allocDeviceMemory(
    data_t **dataTable, data_t **dataBuffer, sample_t **samples, uint_t **ranksEven, uint_t **ranksOdd,
    uint_t tableLen
);
void freeDeviceMemory(
    data_t *dataTable, data_t *dataBuffer, sample_t *samples, uint_t *ranksEven, uint_t *ranksOdd
);

#endif
