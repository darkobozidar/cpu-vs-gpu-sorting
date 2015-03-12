#include <stdio.h>
#include <math.h>
#include <Windows.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"
#include "../Utils/cuda.h"
#include "../Utils/host.h"
#include "constants.h"
#include "data_types.h"
#include "kernels_common.h"
#include "sort.h"


/*
Executes kernel for finding min/max values. Every thread block searches for min/max values in their
corresponding chunk of data. This means kernel will return a list of min/max values with same length
as number of thread blocks executing in kernel.
*/
template <uint_t threadsReduction, uint_t elemsThreadReduction>
uint_t QuicksortParallel::runMinMaxReductionKernel(data_t *d_keys, data_t *d_keysBuffer, uint_t arrayLength)
{
    // Half of the array for min values and the other half for max values
    uint_t sharedMemSize = 2 * threadsReduction * sizeof(*d_keys);
    dim3 dimGrid((arrayLength - 1) / (threadsReduction * elemsThreadReduction) + 1, 1, 1);
    dim3 dimBlock(threadsReduction, 1, 1);

    minMaxReductionKernel<threadsReduction, elemsThreadReduction><<<dimGrid, dimBlock, sharedMemSize>>>(
        d_keys, d_keysBuffer, arrayLength
    );

    return dimGrid.x;
}


/*
Searches for min/max values in array.
*/
template <uint_t thresholdReduction, uint_t threadsReduction, uint_t elemsThreadReduction>
void QuicksortParallel::minMaxReduction(
    data_t *h_keys, data_t *d_keys, data_t *d_keysBuffer, data_t *h_minMaxValues, uint_t arrayLength,
    data_t &minVal, data_t &maxVal
)
{
    minVal = MAX_VAL;
    maxVal = MIN_VAL;

    // Checks whether array is short enough to be reduced entirely on host or if reduction on device is needed
    if (arrayLength > thresholdReduction)
    {
        // Kernel returns array with min/max values of length numVales
        uint_t numValues = runMinMaxReductionKernel<threadsReduction, elemsThreadReduction>(
            d_keys, d_keysBuffer, arrayLength
        );

        cudaError_t error = cudaMemcpy(
            h_minMaxValues, d_keysBuffer, 2 * numValues * sizeof(*h_minMaxValues), cudaMemcpyDeviceToHost
        );
        checkCudaError(error);

        data_t *minValues = h_minMaxValues;
        data_t *maxValues = h_minMaxValues + numValues;

        // Finnishes reduction on host
        for (uint_t i = 0; i < numValues; i++)
        {
            minVal = min(minVal, minValues[i]);
            maxVal = max(maxVal, maxValues[i]);
        }
    }
    else
    {
        for (uint_t i = 0; i < arrayLength; i++)
        {
            minVal = min(minVal, h_keys[i]);
            maxVal = max(maxVal, h_keys[i]);
        }
    }
}

template void QuicksortParallel::minMaxReduction
<THRESHOLD_REDUCTION_KO, THREADS_PER_REDUCTION_KO, ELEMENTS_PER_THREAD_REDUCTION_KO>(
    data_t *h_keys, data_t *d_keys, data_t *d_keysBuffer, data_t *h_minMaxValues, uint_t arrayLength,
    data_t &minVal, data_t &maxVal
);
template void QuicksortParallel::minMaxReduction
<THRESHOLD_REDUCTION_KV, THREADS_PER_REDUCTION_KV, ELEMENTS_PER_THREAD_REDUCTION_KV>(
    data_t *h_keys, data_t *d_keys, data_t *d_keysBuffer, data_t *h_minMaxValues, uint_t arrayLength,
    data_t &minVal, data_t &maxVal
);
