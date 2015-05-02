#ifndef MERGE_SORT_SEQUENTIAL_H
#define MERGE_SORT_SEQUENTIAL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../../Utils/data_types_common.h"
#include "../../Utils/sort_interface.h"
#include "../../Utils/host.h"


/*
Class for sequential merge sort.
*/
class MergeSortSequential : public SortSequential
{
protected:
    std::string _sortName = "Merge sort sequential";

    // Buffer for keys
    data_t *_h_keysBuffer = NULL;
    // Buffer for values
    data_t *_h_valuesBuffer = NULL;

    /*
    Method for allocating memory needed both for key only and key-value sort.
    */
    virtual void memoryAllocate(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        SortSequential::memoryAllocate(h_keys, h_values, arrayLength);

        _h_keysBuffer = (data_t*)malloc(arrayLength * sizeof(*_h_keysBuffer));
        checkMallocError(_h_keysBuffer);
        _h_valuesBuffer = (data_t*)malloc(arrayLength * sizeof(*_h_valuesBuffer));
        checkMallocError(_h_valuesBuffer);
    }

    /*
    In case if number of phases of merge sort is odd, then sorted array is located in buffer array. In that case
    array has to be copied from buffer to primary array.
    This could be also achieved with passing of pointers by reference, but this was easier to implement with
    current class structure.
    */
    virtual void memoryCopyAfterSort(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        SortSequential::memoryCopyAfterSort(h_keys, h_values, arrayLength);

        uint_t numSortPhases = log2(nextPowerOf2(arrayLength));
        if (numSortPhases % 2 == 0)
        {
            return;
        }

        std::copy(_h_keysBuffer, _h_keysBuffer + arrayLength, h_keys);
        if (h_values != NULL)
        {
            std::copy(_h_valuesBuffer, _h_valuesBuffer + arrayLength, h_values);
        }
    }

    /*
    From provided array offset, size of array block and length of entire array returns end index of the block.
    */
    uint_t getEndIndex(uint_t offset, uint_t subBlockSize, uint_t arrayLength)
    {
        uint_t endIndex = offset + subBlockSize;
        return endIndex <= arrayLength ? endIndex : arrayLength;
    }

    /*
    Returns the output array. Implemented because sample sort class (which is derived from this class) requires
    different function to determine output array than this class.
    */
    virtual data_t* getOutputMergeArray(data_t *arrayBuffer, data_t *arraySorted, bool isLastMergePhase)
    {
        return arrayBuffer;
    }

    /*
    Merges two blocks in array and outputs the result to buffer array.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void mergeBlocks(
        data_t *h_keys, data_t *h_values, data_t *h_keysBuffer, data_t *h_valuesBuffer, data_t *h_keysSorted,
        data_t *h_valuesSorted, uint_t arrayLength, uint_t sortedBlockSize, uint_t blockIndex, bool isLastMergePhase
    )
    {
        // Number of sub-blocks being merged
        uint_t subBlockSize = sortedBlockSize / 2;
        // If it is last phase of merge sort, data is copied to result array
        data_t *keysOutput = getOutputMergeArray(h_keysBuffer, h_keysSorted, isLastMergePhase);
        data_t *valuesOutput = getOutputMergeArray(h_valuesBuffer, h_valuesSorted, isLastMergePhase);

        // Odd (left) block being merged
        uint_t oddIndex = blockIndex * sortedBlockSize;
        uint_t oddEnd = getEndIndex(oddIndex, subBlockSize, arrayLength);

        // If there is only odd block without even block, then only odd block is copied into buffer
        if (oddEnd == arrayLength)
        {
            std::copy(h_keys + oddIndex, h_keys + oddEnd, keysOutput + oddIndex);
            if (!sortingKeyOnly)
            {
                std::copy(h_values + oddIndex, h_values + oddEnd, valuesOutput + oddIndex);
            }
            return;
        }

        // Even (right) block being merged
        uint_t evenIndex = oddIndex + subBlockSize;
        uint_t evenEnd = getEndIndex(evenIndex, subBlockSize, arrayLength);
        uint_t mergeIndex = oddIndex;

        // Merge of odd and even block
        while (oddIndex < oddEnd && evenIndex < evenEnd)
        {
            data_t oddElement = h_keys[oddIndex];
            data_t evenElement = h_keys[evenIndex];

            if (sortOrder == ORDER_ASC ? oddElement <= evenElement : oddElement >= evenElement)
            {
                keysOutput[mergeIndex] = oddElement;
                if (!sortingKeyOnly)
                {
                    valuesOutput[mergeIndex] = h_values[oddIndex];
                }

                mergeIndex++;
                oddIndex++;
            }
            else
            {
                keysOutput[mergeIndex] = evenElement;
                if (!sortingKeyOnly)
                {
                    valuesOutput[mergeIndex] = h_values[evenIndex];
                }

                mergeIndex++;
                evenIndex++;
            }
        }

        // Block that wasn't merged entirely is copied into buffer array
        if (oddIndex == oddEnd)
        {
            std::copy(h_keys + evenIndex, h_keys + evenEnd, keysOutput + mergeIndex);
            if (!sortingKeyOnly)
            {
                std::copy(h_values + evenIndex, h_values + evenEnd, valuesOutput + mergeIndex);
            }
        }
        else
        {
            std::copy(h_keys + oddIndex, h_keys + oddEnd, keysOutput + mergeIndex);
            if (!sortingKeyOnly)
            {
                std::copy(h_values + oddIndex, h_values + oddEnd, valuesOutput + mergeIndex);
            }
        }
    }

    /*
    Sorts data sequentially with merge sort.
    Arrays for sorted keys and values are needed in sample sort, which is derived from this class.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void mergeSortSequential(
        data_t *h_keys, data_t *h_values, data_t *h_keysBuffer, data_t *h_valuesBuffer, data_t *h_keysSorted,
        data_t *h_valuesSorted, uint_t arrayLength
    )
    {
        if (arrayLength == 1)
        {
            h_keysSorted[0] = h_keys[0];
            if (!sortingKeyOnly)
            {
                h_valuesSorted[0] = h_values[0];
            }
            return;
        }

        uint_t arrayLenPower2 = nextPowerOf2(arrayLength);

        // Log(arrayLength) phases of merge sort
        for (uint_t sortedBlockSize = 2; sortedBlockSize <= arrayLenPower2; sortedBlockSize *= 2)
        {
            // Number of merged blocks that will be created in this iteration
            uint_t numBlocks = (arrayLength - 1) / sortedBlockSize + 1;
            bool isLastMergePhase = numBlocks == 1;

            // Merge of all blocks
            for (uint_t blockIndex = 0; blockIndex < numBlocks; blockIndex++)
            {
                mergeBlocks<sortOrder, sortingKeyOnly>(
                    h_keys, h_values, h_keysBuffer, h_valuesBuffer, h_keysSorted, h_valuesSorted, arrayLength,
                    sortedBlockSize, blockIndex, isLastMergePhase
                );
            }

            // Exchanges key and value pointers with buffer
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
    Wrapper for merge sort method.
    The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
    */
    void sortKeyOnly()
    {
        if (_sortOrder == ORDER_ASC)
        {
            mergeSortSequential<ORDER_ASC, true>(_h_keys, NULL, _h_keysBuffer, NULL, NULL, NULL, _arrayLength);
        }
        else
        {
            mergeSortSequential<ORDER_DESC, true>(_h_keys, NULL, _h_keysBuffer, NULL, NULL, NULL, _arrayLength);
        }
    }

    /*
    Wrapper for merge sort method.
    The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
    */
    void sortKeyValue()
    {
        if (_sortOrder == ORDER_ASC)
        {
            mergeSortSequential<ORDER_ASC, false>(
                _h_keys, _h_values, _h_keysBuffer, _h_valuesBuffer, NULL, NULL, _arrayLength
            );
        }
        else
        {
            mergeSortSequential<ORDER_DESC, false>(
                _h_keys, _h_values, _h_keysBuffer, _h_valuesBuffer, NULL, NULL, _arrayLength
            );
        }
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
    }

public:
    std::string getSortName()
    {
        return this->_sortName;
    }
};

#endif
