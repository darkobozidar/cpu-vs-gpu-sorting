#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"
#include "constants.h"
#include "data_types.h"




template <order_t sortOrder, direct_t direction>
void sampleSort(data_t *dataTable, data_t *dataBuffer, data_t *samples, uint_t tableLen)
{
    if (tableLen < SMALL_SORT_THRESHOLD)
    {
        // TODO sort
    }

    for (uint_t i = 0; i < NUM_SAMPLES_SEQUENTIAL; i++)
    {
        samples[i] = dataTable[rand() % tableLen];
    }
}

/*
Sorts data sequentially with sample sort.
*/
double sortSequential(
    data_t *dataInput, data_t *dataOutput, data_t *samples, uint_t *bucketSizes, uint_t *elementBuckets,
    uint_t tableLen, order_t sortOrder
)
{
    LARGE_INTEGER timer;
    startStopwatch(&timer);

    // TODO implement

    /*return endStopwatch(timer);*/
    return 9999;
}
