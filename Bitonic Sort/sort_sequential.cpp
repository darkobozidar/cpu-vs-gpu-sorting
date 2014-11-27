#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"


/*
Sorts data sequentially with NORMALIZED bitonic sort.
*/
double sortSequential(data_t* dataTable, uint_t tableLen, order_t sortOrder)
{
    LARGE_INTEGER timer;
    startStopwatch(&timer);

    for (uint_t subBlockSize = 1; subBlockSize < tableLen; subBlockSize <<= 1)
    {
        for (uint_t stride = subBlockSize; stride > 0; stride >>= 1)
        {
            for (uint_t el = 0; el < tableLen >> 1; el++)
            {
                uint_t indexEl = el;
                uint_t offset = stride;

                if (stride == subBlockSize)
                {
                    indexEl = (el / stride) * stride + ((stride - 1) - (el % stride));
                    offset = ((el & (stride - 1)) << 1) + 1;
                }

                uint_t index = (indexEl << 1) - (indexEl & (stride - 1));
                if (index + offset >= tableLen)
                {
                    break;
                }

                if ((dataTable[index] > dataTable[index + offset]) ^ sortOrder)
                {
                    data_t temp = dataTable[index];
                    dataTable[index] = dataTable[index + offset];
                    dataTable[index + offset] = temp;
                }
            }
        }
    }

    return endStopwatch(timer);
}

/*
Compare function for ASCENDING order needed for C++ sort.
*/
int compareAsc(const void* elem1, const void* elem2)
{
    return *((data_t*)elem1) - *((data_t*)elem2);
}

/*
Compare function for DESCENDING order needed for C++ sort.
*/
int compareDesc(const void* elem1, const void* elem2)
{
    return *((data_t*)elem2) - *((data_t*)elem1);
}

/*
Sorts data with C++ sort, which sorts data 100% correctly. This is needed to verify parallel and sequential sorts.
// TODO sort order
*/
double sortCorrect(data_t *dataTable, uint_t tableLen, order_t sortOrder)
{
    LARGE_INTEGER timer;

    startStopwatch(&timer);

    if (sortOrder == ORDER_ASC)
    {
        qsort(dataTable, tableLen, sizeof(*dataTable), compareAsc);
    }
    else
    {
        qsort(dataTable, tableLen, sizeof(*dataTable), compareDesc);
    }

    return endStopwatch(timer);
}
