#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"
#include "sort_sequential.h"


/*
Exchanges elements on provided pointer adresses.
*/
void QuicksortSequentialKeyValue::exchangeElemens(data_t *elem1, data_t *elem2)
{
    data_t temp = *elem1;
    *elem1 = *elem2;
    *elem2 = temp;
}

/*
Searches for pivot.
Searches for median of first, middle and last element in array.
*/
uint_t QuicksortSequentialKeyValue::getPivotIndex(data_t *dataArray, uint_t length)
{
    uint_t index1 = 0;
    uint_t index2 = length / 2;
    uint_t index3 = length - 1;

    if (dataArray[index1] > dataArray[index2])
    {
        if (dataArray[index2] > dataArray[index3])
        {
            return index2;
        }
        else if (dataArray[index1] > dataArray[index3])
        {
            return index3;
        }
        else
        {
            return index1;
        }
    }
    else
    {
        if (dataArray[index1] > dataArray[index3])
        {
            return index1;
        }
        else if (dataArray[index2] > dataArray[index3])
        {
            return index3;
        }
        else
        {
            return index2;
        }
    }
}

/*
Partitions array into 2 partitions - elemens lower and elements greater than pivot.
*/
template <order_t sortOrderTempl>
uint_t QuicksortSequentialKeyValue::partitionArray(data_t *keys, data_t *values, uint_t length)
{
    uint_t pivotIndex = getPivotIndex(keys, length);
    data_t pivotValue = keys[pivotIndex];

    exchangeElemens(&keys[pivotIndex], &keys[length - 1]);
    exchangeElemens(&values[pivotIndex], &values[length - 1]);
    uint_t storeIndex = 0;

    for (uint_t i = 0; i < length - 1; i++)
    {
        if (sortOrderTempl ^ (keys[i] < pivotValue))
        {
            exchangeElemens(&keys[i], &keys[storeIndex]);
            exchangeElemens(&values[i], &values[storeIndex]);
            storeIndex++;
        }
    }

    exchangeElemens(&keys[storeIndex], &keys[length - 1]);
    exchangeElemens(&values[storeIndex], &values[length - 1]);
    return storeIndex;
}

/*
Sorts the array with quicksort.
*/
template <order_t sortOrderTempl>
void QuicksortSequentialKeyValue::quickSort(data_t *keys, data_t *values, uint_t length)
{
    if (length <= 1)
    {
        return;
    }
    else if (length == 2)
    {
        if (sortOrderTempl ^ (keys[0] > keys[1]))
        {
            exchangeElemens(&keys[0], &keys[1]);
            exchangeElemens(&values[0], &values[1]);
        }
        return;
    }

    uint_t partition = partitionArray<sortOrderTempl>(keys, values, length);
    quickSort<sortOrderTempl>(keys, values, partition);
    quickSort<sortOrderTempl>(keys + partition + 1, values + partition + 1, length - partition - 1);
}


/*
Sorts data sequentially with quicksort.

Because of recursion calls static functions.
*/
void QuicksortSequentialKeyValue::sortPrivate()
{
    if (this->sortOrder == ORDER_ASC)
    {
        quickSort<ORDER_ASC>(this->h_keys, this->h_values, this->arrayLength);
    }
    else
    {
        quickSort<ORDER_DESC>(this->h_keys, this->h_values, this->arrayLength);
    }
}
