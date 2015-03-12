#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../Utils/data_types_common.h"
#include "sort_sequential.h"

/*
Sorts data sequentially with NORMALIZED bitonic sort.
*/
template <order_t sortOrder>
void BitonicSortSequentialKeyOnly::bitonicSortSequentialKeyOnly(data_t *h_keys, uint_t arrayLength)
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

                // In normalized bitonic sort, first STEP of every PHASE uses different offset than all other STEPS.
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
                }
            }
        }
    }
}

/*
Wrapper for bitonic sort method.
The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
*/
void BitonicSortSequentialKeyOnly::sortPrivate()
{
    if (_sortOrder == ORDER_ASC)
    {
        bitonicSortSequentialKeyOnly<ORDER_ASC>(_h_keys, _arrayLength);
    }
    else
    {
        bitonicSortSequentialKeyOnly<ORDER_DESC>(_h_keys, _arrayLength);
    }
}
