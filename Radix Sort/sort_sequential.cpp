#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"


/*
Sorts data sequentially with radix sort.
*/
double sortSequential(data_t *dataInput, data_t *dataOutput, uint_t *dataCounters, uint_t tableLen, order_t sortOrder)
{
    LARGE_INTEGER timer;
    uint_t interval = MAX_VAL + 1;

    startStopwatch(&timer);

    // Resets counters
    for (uint_t i = 0; i < interval; i++)
    {
        dataCounters[i] = 0;
    }

    // Counts number of element occurances
    for (uint_t i = 0; i < tableLen; i++)
    {
        dataCounters[dataInput[i]]++;
    }

    // Performs scan on counters
    for (uint_t i = 1; i < interval; i++)
    {
        dataCounters[i] += dataCounters[i - 1];
    }

    for (int_t i = tableLen - 1; i >= 0; i--)
    {
        dataOutput[--dataCounters[dataInput[i]]] = dataInput[i];
    }

    return endStopwatch(timer);
}
