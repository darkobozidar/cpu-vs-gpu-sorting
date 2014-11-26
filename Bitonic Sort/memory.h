#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

#include "../Utils/data_types_common.h"


void allocHostMemory(data_t **input, data_t **outputParallel, data_t **outputCorrect, uint_t tableLen);
void freeHostMemory(data_t *input, data_t *outputParallel, data_t *outputCorrect);

#endif
