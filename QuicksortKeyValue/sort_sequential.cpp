#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"
#include "sort_sequential.h"


/*
Exchanges elements on provided pointer adresses.
*/
void exchangeElemens(data_t *elem1, data_t *elem2)
{
    data_t temp = *elem1;
    *elem1 = *elem2;
    *elem2 = temp;
}

/*
Searches for pivot.
Searches for median of first, middle and last element in array.
*/
uint_t getPivotIndex(data_t *dataTable, uint_t length)
{
    uint_t index1 = 0;
    uint_t index2 = length / 2;
    uint_t index3 = length - 1;

    if (dataTable[index1] > dataTable[index2])
    {
        if (dataTable[index2] > dataTable[index3])
        {
            return index2;
        }
        else if (dataTable[index1] > dataTable[index3])
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
        if (dataTable[index1] > dataTable[index3])
        {
            return index1;
        }
        else if (dataTable[index2] > dataTable[index3])
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
template <order_t sortOrder>
uint_t partitionArray(data_t *dataKeys, data_t *dataValues, uint_t length)
{
    uint_t pivotIndex = getPivotIndex(dataKeys, length);
    data_t pivotValue = dataKeys[pivotIndex];

    exchangeElemens(&dataKeys[pivotIndex], &dataKeys[length - 1]);
    exchangeElemens(&dataValues[pivotIndex], &dataValues[length - 1]);
    uint_t storeIndex = 0;

    for (uint_t i = 0; i < length - 1; i++)
    {
        if (sortOrder ^ (dataKeys[i] < pivotValue))
        {
            exchangeElemens(&dataKeys[i], &dataKeys[storeIndex]);
            exchangeElemens(&dataValues[i], &dataValues[storeIndex]);
            storeIndex++;
        }
    }

    exchangeElemens(&dataKeys[storeIndex], &dataKeys[length - 1]);
    exchangeElemens(&dataValues[storeIndex], &dataValues[length - 1]);
    return storeIndex;
}

/*
Sorts the array with quicksort.
*/
template <order_t sortOrder>
void quickSort(data_t *dataKeys, data_t *dataValues, uint_t length)
{
    if (length <= 1)
    {
        return;
    }
    else if (length == 2)
    {
        if (sortOrder ^ (dataKeys[0] > dataKeys[1]))
        {
            exchangeElemens(&dataKeys[0], &dataKeys[1]);
            exchangeElemens(&dataValues[0], &dataValues[1]);
        }
        return;
    }

    uint_t partition = partitionArray<sortOrder>(dataKeys, dataValues, length);
    quickSort<sortOrder>(dataKeys, dataValues, partition);
    quickSort<sortOrder>(dataKeys + partition + 1, dataValues + partition + 1, length - partition - 1);
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
