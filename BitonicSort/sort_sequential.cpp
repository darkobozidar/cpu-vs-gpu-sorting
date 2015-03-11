#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../Utils/data_types_common.h"
#include "sort_sequential.h"


/*
Sorts data sequentially with NORMALIZED bitonic sort.
*/
void BitonicSortSequentialKeyOnly::sortPrivate()
{
    for (uint_t subBlockSize = 1; subBlockSize < this->arrayLength; subBlockSize <<= 1)
    {
        for (uint_t stride = subBlockSize; stride > 0; stride >>= 1)
        {
            bool isFirstStepOfPhase = stride == subBlockSize;

            for (uint_t el = 0; el < this->arrayLength >> 1; el++)
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
                if (indexRight >= this->arrayLength)
                {
                    break;
                }

                if ((this->h_keys[indexLeft] > this->h_keys[indexRight]) ^ this->sortOrder)
                {
                    data_t temp = this->h_keys[indexLeft];
                    this->h_keys[indexLeft] = this->h_keys[indexRight];
                    this->h_keys[indexRight] = temp;
                }
            }
        }
    }
}
