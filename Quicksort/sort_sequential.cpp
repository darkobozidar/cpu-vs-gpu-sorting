#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"


void exchangeElemens(data_t *elem1, data_t *elem2)
{
    data_t temp = *elem1;
    *elem1 = *elem2;
    *elem2 = temp;
}

uint_t searchMedian(data_t *dataTable, uint_t index1, uint_t index2, uint_t index3)
{
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

data_t getPivotIndex(data_t *dataTable, uint_t length)
{
    if (length <= 2)
    {
        return 0;
    }

    return searchMedian(dataTable, 0, length / 2, length - 1);
}

template <order_t sortOrder>
uint_t partitionArray(data_t *dataTable, uint_t length)
{
    uint_t pivotIndex = getPivotIndex(dataTable, length);
    data_t pivotValue = dataTable[pivotIndex];

    exchangeElemens(&dataTable[pivotIndex], &dataTable[length - 1]);
    uint_t storeIndex = 0;

    for (uint_t i = 0; i < length - 1; i++)
    {
        if (sortOrder ^ (dataTable[i] < pivotValue))
        {
            exchangeElemens(&dataTable[i], &dataTable[storeIndex]);
            storeIndex++;
        }
    }

    exchangeElemens(&dataTable[storeIndex], &dataTable[length - 1]);
    return storeIndex;
}

template <order_t sortOrder>
void quickSort(data_t *dataTable, uint_t length)
{
    if (length <= 1)
    {
        return;
    }

    uint_t partition = partitionArray<sortOrder>(dataTable, length);
    quickSort<sortOrder>(dataTable, partition);
    quickSort<sortOrder>(dataTable + partition + 1, length - partition - 1);
}


/*
Sorts data sequentially with NORMALIZED bitonic sort.
*/
double sortSequential(data_t* dataTable, uint_t tableLen, order_t sortOrder)
{
    LARGE_INTEGER timer;
    startStopwatch(&timer);

    if (sortOrder == ORDER_ASC)
    {
        quickSort<ORDER_ASC>(dataTable, tableLen);
    }
    else
    {
        quickSort<ORDER_DESC>(dataTable, tableLen);
    }

    return endStopwatch(timer);
}
