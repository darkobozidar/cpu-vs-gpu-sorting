#ifndef RADIX_SORT_SEQUENTIAL_H
#define RADIX_SORT_SEQUENTIAL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../../Utils/data_types_common.h"
#include "../../Utils/sort_interface.h"
#include "../../Utils/host.h"
#include "../constants.h"


/*
Parent class for sequential radix sort. Not to be used directly - it's inherited by bottom class, which performs
partial template specialization.
TODO implement for descending order.
*/
template <uint_t bitCountRadixKo, uint_t radixKo, uint_t bitCountRadixKv, uint_t radixKv>
class RadixSortSequentialParent : public SortSequential
{
protected:
    std::string _sortName = "Radix sort sequential";

    // Buffer for keys
    data_t *_h_keysBuffer = NULL;
    // Buffer for values
    data_t *_h_valuesBuffer = NULL;
    // Counters of element occurrences - needed for sequential radix sort
    uint_t *_h_dataCounters;

    /*
    Method for allocating memory needed both for key only and key-value sort.
    */
    virtual void memoryAllocate(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        SortSequential::memoryAllocate(h_keys, h_values, arrayLength);
        uint_t maxRadix = max(radixKo, radixKv);

        // Allocates keys and values
        _h_keysBuffer = (data_t*)malloc(arrayLength * sizeof(*_h_keysBuffer));
        checkMallocError(_h_keysBuffer);
        _h_valuesBuffer = (data_t*)malloc(arrayLength * sizeof(*_h_valuesBuffer));
        checkMallocError(_h_valuesBuffer);
        _h_dataCounters = (uint_t*)malloc(maxRadix * sizeof(*_h_dataCounters));
        checkMallocError(_h_dataCounters);
    }

    /*
    Depending of the number of phases performed by radix sort the sorted array can be located in primary
    or buffer array.
    */
    virtual void memoryCopyAfterSort(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        bool sortingKeyOnly = h_values == NULL;
        uint_t bitCountRadix = sortingKeyOnly ? bitCountRadixKo : bitCountRadixKv;
        uint_t numPhases = DATA_TYPE_BITS / bitCountRadix;

        if (numPhases % 2 == 0)
        {
            SortSequential::memoryCopyAfterSort(h_keys, h_values, arrayLength);
        }
        else
        {
            // Counting sort was performed
            std::copy(_h_keysBuffer, _h_keysBuffer + _arrayLength, h_keys);
            if (!sortingKeyOnly)
            {
                std::copy(_h_valuesBuffer, _h_valuesBuffer + _arrayLength, h_values);
            }
        }
    }

    /*
    Performs sequential counting sort on provided bit offset for specified number of bits.
    */
    template <order_t sortOrder, bool sortingKeyOnly, uint_t radix>
    void countingSort(
        data_t *h_keys, data_t *h_values, data_t *h_keysBuffer, data_t *h_valuesBuffer, uint_t *dataCounters,
        uint_t tableLen, uint_t bitOffset
    )
    {
        // Resets counters
        for (uint_t i = 0; i < radix; i++)
        {
            dataCounters[i] = 0;
        }

        // Counts number of element occurrences
        for (uint_t i = 0; i < tableLen; i++)
        {
            dataCounters[(h_keys[i] >> bitOffset) & (radix - 1)]++;
        }

        // Performs scan on counters
        for (uint_t i = 1; i < radix; i++)
        {
            dataCounters[i] += dataCounters[i - 1];
        }

        // Scatters elements to their output position
        for (int_t i = tableLen - 1; i >= 0; i--)
        {
            uint_t outputIndex = --dataCounters[(h_keys[i] >> bitOffset) & (radix - 1)];

            h_keysBuffer[outputIndex] = h_keys[i];
            if (!sortingKeyOnly)
            {
                h_valuesBuffer[outputIndex] = h_values[i];
            }
        }
    }

    /*
    Sorts data sequentially with radix sort.
    */
    template <order_t sortOrder, bool sortingKeyOnly, uint_t bitCountRadix, uint_t radix>
    void radixSortSequential(
        data_t *h_keys, data_t *h_values, data_t *h_keysBuffer, data_t *h_valuesBuffer, uint_t *dataCounters,
        uint_t arrayLength
    )
    {
        // Executes counting sort for every digit (every group of BIT_COUNT_SEQUENTIAL bits)
        for (uint_t bitOffset = 0; bitOffset < sizeof(data_t)* 8; bitOffset += bitCountRadix)
        {
            countingSort<sortOrder, sortingKeyOnly, radix>(
                h_keys, h_values, h_keysBuffer, h_valuesBuffer, dataCounters, arrayLength, bitOffset
            );

            data_t *temp = h_keys;
            h_keys = h_keysBuffer;
            h_keysBuffer = temp;

            if (!sortingKeyOnly)
            {
                temp = h_values;
                h_values = h_valuesBuffer;
                h_valuesBuffer = temp;
            }
        }
    }

    /*
    Wrapper for bitonic sort method.
    The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
    */
    void sortKeyOnly()
    {
        if (_sortOrder == ORDER_ASC)
        {
            radixSortSequential<ORDER_ASC, true, bitCountRadixKo, radixKo>(
                _h_keys, NULL, _h_keysBuffer, NULL, _h_dataCounters, _arrayLength
            );
        }
        else
        {
            radixSortSequential<ORDER_DESC, true, bitCountRadixKo, radixKo>(
                _h_keys, NULL, _h_keysBuffer, NULL, _h_dataCounters, _arrayLength
            );
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
            radixSortSequential<ORDER_ASC, false, bitCountRadixKv, radixKv>(
                _h_keys, _h_values, _h_keysBuffer, _h_valuesBuffer, _h_dataCounters, _arrayLength
            );
        }
        else
        {
            radixSortSequential<ORDER_DESC, false, bitCountRadixKv, radixKv>(
                _h_keys, _h_values, _h_keysBuffer, _h_valuesBuffer, _h_dataCounters, _arrayLength
            );
        }
    }

public:
    std::string getSortName()
    {
        return this->_sortName;
    }

    /*
    Method for destroying memory needed for sort. For sort testing purposes this method is public.
    */
    void memoryDestroy()
    {
        if (_arrayLength == 0)
        {
            return;
        }

        SortSequential::memoryDestroy();

        free(_h_keysBuffer);
        free(_h_valuesBuffer);
        free(_h_dataCounters);
    }
};

/*
Base class for sequential radix sort with only one template argument for key only and key-value - number of
bits in radix.
*/
template <uint_t bitCountRadixKo, uint_t bitCountRadixKv>
class RadixSortSequentialBase : public RadixSortSequentialParent<
    bitCountRadixKo, 1 << bitCountRadixKo, bitCountRadixKv, 1 << bitCountRadixKv
>
{};

/*
Class for sequential radix sort.
*/
class RadixSortSequential : public RadixSortSequentialBase<BIT_COUNT_SEQUENTIAL_KO, BIT_COUNT_SEQUENTIAL_KV>
{};

#endif
