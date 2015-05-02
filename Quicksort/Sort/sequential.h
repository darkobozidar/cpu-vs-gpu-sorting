#ifndef QUICKSORT_SEQUENTIAL_H
#define QUICKSORT_SEQUENTIAL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../../Utils/data_types_common.h"
#include "../../Utils/sort_interface.h"
#include "../../Utils/host.h"


/*
Class for sequential quicksort.
*/
class QuicksortSequential : public SortSequential
{
protected:
    std::string _sortName = "Quicksort sequential";

    /*
    Exchanges elements on provided pointer adresses.
    */
    void exchangeElemens(data_t *elem0, data_t *elem1)
    {
        data_t temp = *elem0;
        *elem0 = *elem1;
        *elem1 = temp;
    }

    /*
    Searches for pivot - searches for median of first, middle and last element in array.
    */
    uint_t getPivotIndex(data_t *h_keys, uint_t arrayLength)
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

    /*
    Partitions keys into 2 partitions - elements lower and elements greater than pivot.
    */
    template <order_t sortOrder>
    uint_t partitionArray(data_t *h_keys, uint_t arrayLength)
    {
        uint_t pivotIndex = getPivotIndex(h_keys, arrayLength);
        data_t pivotValue = h_keys[pivotIndex];

        exchangeElemens(&h_keys[pivotIndex], &h_keys[arrayLength - 1]);
        uint_t storeIndex = 0;

        for (uint_t i = 0; i < arrayLength - 1; i++)
        {
            if (sortOrder ^ (h_keys[i] <= pivotValue))
            {
                exchangeElemens(&h_keys[i], &h_keys[storeIndex]);
                storeIndex++;
            }
        }

        exchangeElemens(&h_keys[storeIndex], &h_keys[arrayLength - 1]);
        return storeIndex;
    }

    /*
    Partitions keys and values into 2 partitions - elements lower and elements greater than pivot.
    */
    template <order_t sortOrder>
    uint_t partitionArray(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        uint_t pivotIndex = getPivotIndex(h_keys, arrayLength);
        data_t pivotValue = h_keys[pivotIndex];

        exchangeElemens(&h_keys[pivotIndex], &h_keys[arrayLength - 1]);
        exchangeElemens(&h_values[pivotIndex], &h_values[arrayLength - 1]);
        uint_t storeIndex = 0;

        for (uint_t i = 0; i < arrayLength - 1; i++)
        {
            if (sortOrder ^ (h_keys[i] <= pivotValue))
            {
                exchangeElemens(&h_keys[i], &h_keys[storeIndex]);
                exchangeElemens(&h_values[i], &h_values[storeIndex]);
                storeIndex++;
            }
        }

        exchangeElemens(&h_keys[storeIndex], &h_keys[arrayLength - 1]);
        exchangeElemens(&h_values[storeIndex], &h_values[arrayLength - 1]);
        return storeIndex;
    }

    /*
    Sorts keys only with quicksort.
    */
    template <order_t sortOrder>
    void quicksortSequential(data_t *h_keys, uint_t arrayLength)
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
    Sorts key-value pairs with quicksort.
    */
    template <order_t sortOrder>
    void quicksortSequential(data_t *h_keys, data_t *h_values, uint_t arrayLength)
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
                exchangeElemens(&h_values[0], &h_values[1]);
            }
            return;
        }

        uint_t partition = partitionArray<sortOrder>(h_keys, h_values, arrayLength);
        quicksortSequential<sortOrder>(h_keys, h_values, partition);
        quicksortSequential<sortOrder>(h_keys + partition + 1, h_values + partition + 1, arrayLength - partition - 1);
    }

    /*
    Wrapper for bitonic sort method.
    The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
    */
    void sortKeyOnly()
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

    /*
    Wrapper for bitonic sort method.
    The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
    */
    void sortKeyValue()
    {
        if (_sortOrder == ORDER_ASC)
        {
            quicksortSequential<ORDER_ASC>(_h_keys, _h_values, _arrayLength);
        }
        else
        {
            quicksortSequential<ORDER_DESC>(_h_keys, _h_values, _arrayLength);
        }
    }

public:
    std::string getSortName()
    {
        return this->_sortName;
    }
};

#endif
