#ifndef BITONIC_SORT_SEQUENTIAL_H
#define BITONIC_SORT_SEQUENTIAL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../../Utils/data_types_common.h"
#include "../../Utils/sort_interface.h"
#include "../../Utils/host.h"


/*
Class for sequential bitonic sort.
*/
class BitonicSortSequential : public SortSequential
{
protected:
    std::string _sortName = "Bitonic sort sequential";

    /*
    Sorts data sequentially with NORMALIZED bitonic sort.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void bitonicSortSequential(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        for (uint_t subBlockSize = 1; subBlockSize < arrayLength; subBlockSize <<= 1)
        {
            for (uint_t stride = subBlockSize; stride > 0; stride >>= 1)
            {
                bool isFirstStepOfPhase = stride == subBlockSize;

                for (uint_t el = 0; el < arrayLength >> 1; el++)
                {
                    uint_t index = el;
                    uint_t offset = stride;

                    // In normalized bitonic sort, first STEP of every PHASE demands different offset than all other STEPS.
                    if (isFirstStepOfPhase)
                    {
                        index = (el / stride) * stride + ((stride - 1) - (el % stride));
                        offset = ((el & (stride - 1)) << 1) + 1;
                    }

                    // Calculates index of left and right element, which are candidates for exchange
                    uint_t indexLeft = (index << 1) - (index & (stride - 1));
                    uint_t indexRight = indexLeft + offset;
                    if (indexRight >= arrayLength)
                    {
                        break;
                    }

                    if ((h_keys[indexLeft] > h_keys[indexRight]) ^ sortOrder)
                    {
                        data_t temp = h_keys[indexLeft];
                        h_keys[indexLeft] = h_keys[indexRight];
                        h_keys[indexRight] = temp;

                        if (!sortingKeyOnly)
                        {
                            temp = h_values[indexLeft];
                            h_values[indexLeft] = h_values[indexRight];
                            h_values[indexRight] = temp;
                        }
                    }
                }
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
            bitonicSortSequential<ORDER_ASC, true>(_h_keys, NULL, _arrayLength);
        }
        else
        {
            bitonicSortSequential<ORDER_DESC, true>(_h_keys, NULL, _arrayLength);
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
            bitonicSortSequential<ORDER_ASC, false>(_h_keys, _h_values, _arrayLength);
        }
        else
        {
            bitonicSortSequential<ORDER_DESC, false>(_h_keys, _h_values, _arrayLength);
        }
    }

public:
    std::string getSortName()
    {
        return this->_sortName;
    }
};

#endif
