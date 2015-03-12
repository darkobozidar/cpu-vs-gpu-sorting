#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"
#include "sort.h"


/*
Exchanges elements on provided pointer adresses.
*/
void QuicksortSequential::exchangeElemens(data_t *elem0, data_t *elem1)
{
    data_t temp = *elem0;
    *elem0 = *elem1;
    *elem1 = temp;
}

/*
Searches for pivot.
Searches for median of first, middle and last element in array.
*/
uint_t QuicksortSequential::getPivotIndex(data_t *h_keys, uint_t arrayLength)
{
    uint_t index1 = 0;
    uint_t index2 = arrayLength / 2;
    uint_t index3 = arrayLength - 1;

    if (h_keys[index1] > h_keys[index2])
    {
        if (h_keys[index2] > h_keys[index3])
        {
            return index2;
        }
        else if (h_keys[index1] > h_keys[index3])
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
        if (h_keys[index1] > h_keys[index3])
        {
            return index1;
        }
        else if (h_keys[index2] > h_keys[index3])
        {
            return index3;
        }
        else
        {
            return index2;
        }
    }
}
