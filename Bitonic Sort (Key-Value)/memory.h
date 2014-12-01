#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

#include "../Utils/data_types_common.h"


void allocHostMemory(
    data_t **inputKeys, data_t **inputValues, data_t **outputParallelKeys, data_t **outputParallelValues,
    data_t **outputSequentialKeys, data_t **outputSequentialValues, data_t **outputCorrect, double ***timers,
    uint_t tableLen, uint_t testRepetitions
);
void freeHostMemory(
    data_t *inputKeys, data_t *inputValues, data_t *outputParallelKeys, data_t *outputParallelValues,
    data_t *outputSequentialKeys, data_t *outputSequentialValues, data_t *outputCorrect, double **timers
);

void allocDeviceMemory(data_t **dataTableKeys, data_t **dataTableValues, uint_t tableLen);
void freeDeviceMemory(data_t *dataTableKeys, data_t *dataTableValues);

#endif
