#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"
#include "constants.h"


/*
Performs sequential couinting sort on provided bit offset for specified number of bits.
*/
void countingSort(
    data_t *keysInput, data_t *valuesInput, data_t *keysOutput, data_t *valuesOutput, uint_t *dataCounters,
    uint_t tableLen, uint_t bitOffset, order_t sortOrder
)
{
    // Resets counters
    for (uint_t i = 0; i < RADIX_SEQUENTIAL; i++)
    {
        dataCounters[i] = 0;
    }

    // Counts number of element occurances
    for (uint_t i = 0; i < tableLen; i++)
    {
        dataCounters[(keysInput[i] >> bitOffset) & RADIX_MASK_SEQUENTIAL]++;
    }

    // Performs scan on counters
    for (uint_t i = 1; i < RADIX_SEQUENTIAL; i++)
    {
        dataCounters[i] += dataCounters[i - 1];
    }

    // Scatters elements to their output position
    for (int_t i = tableLen - 1; i >= 0; i--)
    {
        uint_t outputIndex = --dataCounters[(keysInput[i] >> bitOffset) & RADIX_MASK_SEQUENTIAL];
        keysOutput[outputIndex] = keysInput[i];
        valuesOutput[outputIndex] = valuesInput[i];
    }
}

/*
Sorts data sequentially with radix sort.
*/
double sortSequential(
    data_t *&keysInput, data_t *&valuesInput, data_t *&keysOutput, data_t *&valuesOutput, uint_t *dataCounters,
    uint_t tableLen, order_t sortOrder
)
{
    LARGE_INTEGER timer;
    startStopwatch(&timer);

    // Executes couting sort for every digit (every group of BIT_COUNT_SEQUENTIAL bits)
    for (uint_t bitOffset = 0; bitOffset < sizeof(data_t) * 8; bitOffset += BIT_COUNT_SEQUENTIAL)
    {
        countingSort(
            keysInput, valuesInput, keysOutput, valuesOutput, dataCounters, tableLen, bitOffset, sortOrder
        );

        data_t *temp = keysInput;
        keysInput = keysOutput;
        keysOutput = temp;
    }

    data_t *temp = keysInput;
    keysInput = keysOutput;
    keysOutput = temp;

    return endStopwatch(timer);
}
