#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"


/*
Sorts data sequentially with NORMALIZED bitonic sort.
*/
double sortSequential(data_t *dataTable, uint_t tableLen, order_t sortOrder)
{
    LARGE_INTEGER timer;
    startStopwatch(&timer);

    for (uint_t subBlockSize = 1; subBlockSize < tableLen; subBlockSize <<= 1)
    {
        for (uint_t stride = subBlockSize; stride > 0; stride >>= 1)
        {
            bool isFirstStepOfPhase = stride == subBlockSize;

            for (uint_t el = 0; el < tableLen >> 1; el++)
            {
                uint_t index = el;
                uint_t offset = stride;

                // In normalized bitonic sort, first STEP of every PHASE uses different offset than all other STEPS.
                if (isFirstStepOfPhase)
                {
                    index = (el / stride) * stride + ((stride - 1) - (el % stride));
                    offset = ((el & (stride - 1)) << 1) + 1;
                }

                // Calculates index of left and right element, which are candidates for exchange
                uint_t indexLeft = (index << 1) - (index & (stride - 1));
                uint_t indexRight = indexLeft + offset;
                if (indexRight >= tableLen)
                {
                    break;
                }

                if ((dataTable[indexLeft] > dataTable[indexRight]) ^ sortOrder)
                {
                    data_t temp = dataTable[indexLeft];
                    dataTable[indexLeft] = dataTable[indexRight];
                    dataTable[indexRight] = temp;
                }
            }
        }
    }

    return endStopwatch(timer);
}
