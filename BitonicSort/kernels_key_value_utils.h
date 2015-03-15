#ifndef KERNEL_KEY_VALUE_UTILS_BITONIC_SORT_H
#define KERNEL_KEY_VALUE_UTILS_BITONIC_SORT_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"
#include "../Utils/kernel.h"


template <order_t sortOrder, uint_t threadsKernel, uint_t elemsKernel, bool isFirstStepOfPhase>
inline __device__ void bitonicMergeStep(
    data_t *keys, data_t *values, uint_t offsetGlobal, uint_t tableLen, uint_t stride
)
{
    // Every thread compares and exchanges 2 elements
    for (uint_t tx = threadIdx.x; tx < (threadsKernel * elemsKernel) >> 1; tx += threadsKernel)
    {
        uint_t indexThread = offsetGlobal + tx;
        uint_t offset = stride;

        // In NORMALIZED bitonic sort, first STEP of every PHASE uses different offset than all other
        // STEPS. Also, in first step of every phase, offset sizes are generated in ASCENDING order
        // (normalized bitnic sort requires DESCENDING order). Because of that, we can break the loop if
        // index + offset >= length (bellow). If we want to generate offset sizes in ASCENDING order,
        // than thread indexes inside every sub-block have to be reversed.
        if (isFirstStepOfPhase)
        {
            offset = ((indexThread & (stride - 1)) << 1) + 1;
            indexThread = (indexThread / stride) * stride + ((stride - 1) - (indexThread % stride));
        }

        uint_t index = (indexThread << 1) - (indexThread & (stride - 1));
        if (index + offset >= tableLen)
        {
            break;
        }

        compareExchange<sortOrder>(
            &keys[index], &keys[index + offset], &values[index], &values[index + offset]
        );
    }
}

/*
Sorts data with NORMALIZED bitonic sort.
*/
template <order_t sortOrder, uint_t threadsBitonicSort, uint_t elemsBitonicSort>
inline __device__ void normalizedBitonicSort(
    data_t *keysInput, data_t *valuesInput, data_t *keysOutput, data_t *valuesOutput, uint_t tableLen
)
{
    extern __shared__ data_t bitonicSortTile[];

    uint_t elemsPerThreadBlock = threadsBitonicSort * elemsBitonicSort;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    uint_t dataBlockLength = offset + elemsPerThreadBlock <= tableLen ? elemsPerThreadBlock : tableLen - offset;

    data_t *keysTile = bitonicSortTile;
    data_t *valuesTile = bitonicSortTile + dataBlockLength;

    // Reads data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += threadsBitonicSort)
    {
        keysTile[tx] = keysInput[offset + tx];
        valuesTile[tx] = valuesInput[offset + tx];
    }
    __syncthreads();

    // Bitonic sort PHASES
    for (uint_t subBlockSize = 1; subBlockSize < dataBlockLength; subBlockSize <<= 1)
    {
        // Bitonic merge STEPS
        for (uint_t stride = subBlockSize; stride > 0; stride >>= 1)
        {
            if (stride == subBlockSize)
            {
                bitonicMergeStep<sortOrder, threadsBitonicSort, elemsBitonicSort, true>(
                    keysTile, valuesTile, 0, dataBlockLength, stride
                );
            }
            else
            {
                bitonicMergeStep<sortOrder, threadsBitonicSort, elemsBitonicSort, false>(
                    keysTile, valuesTile, 0, dataBlockLength, stride
                );
            }

            __syncthreads();
        }
    }

    // Stores data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += threadsBitonicSort)
    {
        keysOutput[offset + tx] = keysTile[tx];
        valuesOutput[offset + tx] = valuesTile[tx];
    }
}

/*
Local bitonic merge for sections, where stride IS LOWER OR EQUAL than max shared memory.
*/
template <order_t sortOrder, bool isFirstStepOfPhase, uint_t threadsMerge, uint_t elemsMerge>
inline __device__ void bitonicMergeLocal(data_t *keys, data_t *values, uint_t tableLen, uint_t step)
{
    extern __shared__ data_t mergeTile[];
    bool isFirstStepOfPhaseCopy = isFirstStepOfPhase;  // isFirstStepOfPhase is not editable (constant)

    uint_t elemsPerThreadBlock = threadsMerge * elemsMerge;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    uint_t dataBlockLength = offset + elemsPerThreadBlock <= tableLen ? elemsPerThreadBlock : tableLen - offset;

    data_t *keysTile = mergeTile;
    data_t *valuesTile = mergeTile + dataBlockLength;

    // Reads data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += threadsMerge)
    {
        keysTile[tx] = keys[offset + tx];
        valuesTile[tx] = values[offset + tx];
    }
    __syncthreads();

    // Bitonic merge
    for (uint_t stride = 1 << (step - 1); stride > 0; stride >>= 1)
    {
        if (isFirstStepOfPhaseCopy)
        {
            bitonicMergeStep<sortOrder, threadsMerge, elemsMerge, true>(
                keysTile, valuesTile, 0, dataBlockLength, stride
            );
            isFirstStepOfPhaseCopy = false;
        }
        else
        {
            bitonicMergeStep<sortOrder, threadsMerge, elemsMerge, false>(
                keysTile, valuesTile, 0, dataBlockLength, stride
            );
        }
        __syncthreads();
    }

    // Stores data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += threadsMerge)
    {
        keys[offset + tx] = keysTile[tx];
        values[offset + tx] = valuesTile[tx];
    }
}

#endif
