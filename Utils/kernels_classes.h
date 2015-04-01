#ifndef KERNELS_CLASSES_H
#define KERNELS_CLASSES_H

#include <stdio.h>
#include <math.h>
#include <Windows.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_types_common.h"

#define __CUDA_INTERNAL_COMPILATION__
#include "kernels.h"
#undef __CUDA_INTERNAL_COMPILATION__


/*
Class which runs kernel for adding padding.
*/
template <uint_t threadsPadding, uint_t elemsPadding>
class AddPaddingBase
{
private:
    /*
    Adds padding of MAX/MIN values to input table, depending if sort order is ascending or descending.
    */
    template <order_t sortOrder, bool fillBuffer>
    void runAddPaddingKernel(data_t *d_arrayPrimary, data_t *d_arrayBuffer, uint_t indexStart, uint_t indexEnd)
    {
        if (indexStart == indexEnd)
        {
            return;
        }

        uint_t paddingLength = indexEnd - indexStart;
        uint_t elemsPerThreadBlock = threadsPadding * elemsPadding;
        dim3 dimGrid((paddingLength - 1) / elemsPerThreadBlock + 1, 1, 1);
        dim3 dimBlock(threadsPadding, 1, 1);

        // Depending on sort order different value is used for padding.
        if (sortOrder == ORDER_ASC)
        {
            addPaddingKernel<threadsPadding, elemsPadding, fillBuffer, MAX_VAL><<<dimGrid, dimBlock>>>(
                d_arrayPrimary, d_arrayBuffer, indexStart, paddingLength
            );
        }
        else
        {
            addPaddingKernel<threadsPadding, elemsPadding, fillBuffer, MIN_VAL><<<dimGrid, dimBlock>>>(
                d_arrayPrimary, d_arrayBuffer, indexStart, paddingLength
            );
        }
    }

protected:
    /*
    Adds padding for primary array only.
    */
    template <order_t sortOrder>
    void runAddPaddingKernel(data_t *d_arrayPrimary, uint_t indexStart, uint_t indexEnd)
    {
        runAddPaddingKernel<sortOrder, false>(d_arrayPrimary, NULL, indexStart, indexEnd);
    }

    /*
    Adds padding for primary and buffer array.
    */
    template <order_t sortOrder>
    void runAddPaddingKernel(data_t *d_arrayPrimary, data_t *d_arrayBuffer, uint_t indexStart, uint_t indexEnd)
    {
        runAddPaddingKernel<sortOrder, true>(d_arrayPrimary, d_arrayBuffer, indexStart, indexEnd);
    }
};

#endif
