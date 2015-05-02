#ifndef KERNELS_COMMON_BITONIC_SORT_ADAPTIVE_H
#define KERNELS_COMMON_BITONIC_SORT_ADAPTIVE_H

#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../../Utils/data_types_common.h"
#include "../../Utils/kernels_utils.h"
#include "../data_types.h"
#include "common_utils.h"


/*
Generates initial intervals and continues to evolve them until the end step.
Note: "blockDim.x" has to be used throughout the entire kernel, because thread block size is variable
*/
template <order_t sortOrder, uint_t elemsInitIntervals>
__global__ void initIntervalsKernel(
    data_t *table, interval_t *intervals, uint_t tableLen, uint_t stepStart, uint_t stepEnd
)
{
    extern __shared__ interval_t intervalsTile[];
    uint_t subBlockSize = 1 << stepStart;
    uint_t activeThreadsPerBlock = tableLen / subBlockSize / gridDim.x;
    uint_t elemsPerThreadBlock = blockDim.x * elemsInitIntervals;

    for (uint_t tx = threadIdx.x; tx < activeThreadsPerBlock; tx += blockDim.x)
    {
        uint_t intervalIndex = blockIdx.x * activeThreadsPerBlock + tx;
        uint_t offset0 = intervalIndex * subBlockSize;
        uint_t offset1 = intervalIndex * subBlockSize + subBlockSize / 2;

        // In every odd block intervals have to be reversed, otherwise intervals aren't generated correctly.
        intervalsTile[tx].offset0 = intervalIndex % 2 ? offset1 : offset0;
        intervalsTile[tx].offset1 = intervalIndex % 2 ? offset0 : offset1;
        intervalsTile[tx].length0 = subBlockSize / 2;
        intervalsTile[tx].length1 = subBlockSize / 2;
    }
    __syncthreads();

    // Evolves intervals in shared memory to end step
    generateIntervals<sortOrder, elemsInitIntervals>(
        table, subBlockSize / 2, 1 << stepEnd, 1, activeThreadsPerBlock
    );

    // Calculates offset in global intervals array
    interval_t *outputIntervalsGlobal = intervals + blockIdx.x * elemsPerThreadBlock;
    // Depending if the number of repetitions is divisible by 2, generated intervals are located in FIRST half
    // OR in SECOND half of shared memory (shared memory has 2x size of generated intervals for buffer purposes)
    interval_t *outputIntervalsLocal = intervalsTile + ((stepStart - stepEnd) % 2 != 0 ? elemsPerThreadBlock : 0);

    // Stores generated intervals from shared to global memory
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += blockDim.x)
    {
        outputIntervalsGlobal[tx] = outputIntervalsLocal[tx];
    }
}

/*
Reads the existing intervals from global memory and evolves them until the end step.
Note: "blockDim.x" has to be used throughout the entire kernel, because thread block size is variable
*/
template <order_t sortOrder, uint_t elemsGenIntervals>
__global__ void generateIntervalsKernel(
    data_t *table, interval_t *inputIntervals, interval_t *outputIntervals, uint_t tableLen, uint_t phase,
    uint_t stepStart, uint_t stepEnd
)
{
    extern __shared__ interval_t intervalsTile[];
    uint_t subBlockSize = 1 << stepStart;
    uint_t activeThreadsPerBlock = tableLen / subBlockSize / gridDim.x;
    interval_t *inputIntervalsGlobal = inputIntervals + blockIdx.x * activeThreadsPerBlock;

    // Active threads read existing intervals from global memory
    for (uint_t tx = threadIdx.x; tx < activeThreadsPerBlock; tx += blockDim.x)
    {
        intervalsTile[tx] = inputIntervalsGlobal[tx];
    }
    __syncthreads();

    // Evolves intervals in shared memory to end step
    generateIntervals<sortOrder, elemsGenIntervals>(
        table, subBlockSize / 2, 1 << stepEnd, 1 << (phase - stepStart), activeThreadsPerBlock
    );

    uint_t elemsPerThreadBlock = blockDim.x * elemsGenIntervals;
    // Calculates offset in global intervals array
    interval_t *outputIntervalsGlobal = outputIntervals + blockIdx.x * elemsPerThreadBlock;
    // Depending if the number of repetitions is divisible by 2, generated intervals are located in FIRST half
    // OR in SECOND half of shared memory (shared memory has 2x size of all generated intervals for buffer purposes)
    interval_t *outputIntervalsLocal = intervalsTile + ((stepStart - stepEnd) % 2 != 0 ? elemsPerThreadBlock : 0);

    // Stores generated intervals from shared to global memory
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += blockDim.x)
    {
        outputIntervalsGlobal[tx] = outputIntervalsLocal[tx];
    }
}

#endif
