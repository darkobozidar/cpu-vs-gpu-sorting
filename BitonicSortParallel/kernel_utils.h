#ifndef KERNEL_UTILS_BITONIC_PARALLEL_H
#define KERNEL_UTILS_BITONIC_PARALLEL_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"


/*
Compares 2 elements and exchanges them according to sortOrder.
*/
template <order_t sortOrder>
inline __device__ void compareExchange(data_t *elem1, data_t *elem2)
{
    if ((*elem1 > *elem2) ^ sortOrder)
    {
        data_t temp = *elem1;
        *elem1 = *elem2;
        *elem2 = temp;
    }
}

/*
Compares 2 elements and exchanges them according to sortOrder.
*/
template <order_t sortOrder>
inline __device__ void compareExchange(data_t *key1, data_t *key2, data_t *val1, data_t *val2)
{
    if ((*key1 > *key2) ^ sortOrder)
    {
        data_t temp = *key1;
        *key1 = *key2;
        *key2 = temp;

        temp = *val1;
        *val1 = *val2;
        *val2 = temp;
    }
}

#endif
