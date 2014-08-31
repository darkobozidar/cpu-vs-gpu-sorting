#ifndef UTILS_DEVICE_H
#define UTILS_DEVICE_H

#include "cuda_runtime.h"
#include "data_types.h"

__device__ void printOnce(char* text, uint_t threadIndex);
__device__ void printOnce(char* text);

#endif
