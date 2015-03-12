#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"
#include "sort.h"


/*
Partitions array into 2 partitions - elemens lower and elements greater than pivot.
*/
template <order_t sortOrder>
uint_t QuicksortSequential::partitionArray(data_t *h_keys, uint_t arrayLength)
{
    uint_t pivotIndex = getPivotIndex(h_keys, arrayLength);
    data_t pivotValue = h_keys[pivotIndex];

    exchangeElemens(&h_keys[pivotIndex], &h_keys[arrayLength - 1]);
    uint_t storeIndex = 0;

    for (uint_t i = 0; i < arrayLength - 1; i++)
    {
        if (sortOrder ^ (h_keys[i] < pivotValue))
        {
            exchangeElemens(&h_keys[i], &h_keys[storeIndex]);
            storeIndex++;
        }
    }

    exchangeElemens(&h_keys[storeIndex], &h_keys[arrayLength - 1]);
    return storeIndex;
}

/*
Sorts the array with quicksort.
*/
template <order_t sortOrder>
void QuicksortSequential::quicksortSequential(data_t *h_keys, uint_t arrayLength)
{
    if (arrayLength <= 1)
    {
        return;
    }
    else if (arrayLength == 2)
    {
        if (sortOrder ^ (h_keys[0] > h_keys[1]))
        {
            exchangeElemens(&h_keys[0], &h_keys[1]);
        }
        return;
    }

    uint_t partition = partitionArray<sortOrder>(h_keys, arrayLength);
    quicksortSequential<sortOrder>(h_keys, partition);
    quicksortSequential<sortOrder>(h_keys + partition + 1, arrayLength - partition - 1);
}

/*
Wrapper for bitonic sort method.
The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
*/
void QuicksortSequential::sortKeyOnly()
{
    if (_sortOrder == ORDER_ASC)
    {
        quicksortSequential<ORDER_ASC>(_h_keys, _arrayLength);
    }
    else
    {
        quicksortSequential<ORDER_DESC>(_h_keys, _arrayLength);
    }
}
