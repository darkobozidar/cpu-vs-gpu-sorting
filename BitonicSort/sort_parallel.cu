#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"
#include "constants.h"
#include "sort_parallel.h"
#include "kernels.h"



/*
Sorts sub-blocks of input data with bitonic sort.
*/
void BitonicSortParallelKeyOnly::runBitoicSortKernel()
{
    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    uint_t sharedMemSize = elemsPerThreadBlock * sizeof(*(this->d_keys));

    dim3 dimGrid((this->arrayLength - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_BITONIC_SORT, 1, 1);

    if (sortOrder == ORDER_ASC)
    {
        bitonicSortKernel<ORDER_ASC><<<dimGrid, dimBlock, sharedMemSize>>>(this->d_keys, this->arrayLength);
    }
    else
    {
        bitonicSortKernel<ORDER_DESC><<<dimGrid, dimBlock, sharedMemSize>>>(this->d_keys, this->arrayLength);
    }
}

/*
Merges array, if data blocks are larger than shared memory size. It executes only of STEP on PHASE per
kernel launch.
*/
void BitonicSortParallelKeyOnly::runBitonicMergeGlobalKernel(uint_t phase, uint_t step)
{
    uint_t elemsPerThreadBlock = THREADS_PER_GLOBAL_MERGE * ELEMS_PER_THREAD_GLOBAL_MERGE;
    dim3 dimGrid((this->arrayLength - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_GLOBAL_MERGE, 1, 1);

    bool isFirstStepOfPhase = phase == step;

    if (sortOrder == ORDER_ASC)
    {
        if (isFirstStepOfPhase)
        {
            bitonicMergeGlobalKernel<ORDER_ASC, true><<<dimGrid, dimBlock>>>(
                this->d_keys, this->arrayLength, step
            );
        }
        else
        {
            bitonicMergeGlobalKernel<ORDER_ASC, false><<<dimGrid, dimBlock>>>(
                this->d_keys, this->arrayLength, step
            );
        }
    }
    else
    {
        if (isFirstStepOfPhase)
        {
            bitonicMergeGlobalKernel<ORDER_DESC, true><<<dimGrid, dimBlock>>>(
                this->d_keys, this->arrayLength, step
            );
        }
        else
        {
            bitonicMergeGlobalKernel<ORDER_DESC, false><<<dimGrid, dimBlock>>>(
                this->d_keys, this->arrayLength, step
            );
        }
    }
}

/*
Merges array when stride is lower than shared memory size. It executes all remaining STEPS of current PHASE.
*/
void BitonicSortParallelKeyOnly::runBitoicMergeLocalKernel(uint_t phase, uint_t step)
{
    // Every thread loads and sorts 2 elements
    uint_t elemsPerThreadBlock = THREADS_PER_LOCAL_MERGE * ELEMS_PER_THREAD_LOCAL_MERGE;
    uint_t sharedMemSize = elemsPerThreadBlock * sizeof(*(this->d_keys));
    dim3 dimGrid((this->arrayLength - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_LOCAL_MERGE, 1, 1);

    bool isFirstStepOfPhase = phase == step;

    if (sortOrder == ORDER_ASC)
    {
        if (isFirstStepOfPhase)
        {
            bitonicMergeLocalKernel<ORDER_ASC, true><<<dimGrid, dimBlock, sharedMemSize>>>(
                this->d_keys, this->arrayLength, step
            );
        }
        else
        {
            bitonicMergeLocalKernel<ORDER_ASC, false><<<dimGrid, dimBlock, sharedMemSize>>>(
                this->d_keys, this->arrayLength, step
            );
        }
    }
    else
    {
        if (isFirstStepOfPhase)
        {
            bitonicMergeLocalKernel<ORDER_DESC, true><<<dimGrid, dimBlock, sharedMemSize>>>(
                this->d_keys, this->arrayLength, step
            );
        }
        else
        {
            bitonicMergeLocalKernel<ORDER_DESC, false><<<dimGrid, dimBlock, sharedMemSize>>>(
                this->d_keys, this->arrayLength, step
            );
        }
    }
}

/*
Sorts data with NORMALIZED BITONIC SORT.
*/
void BitonicSortParallelKeyOnly::sortPrivate()
{
    uint_t tableLenPower2 = nextPowerOf2(this->arrayLength);
    uint_t elemsPerBlockBitonicSort = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    uint_t elemsPerBlockMergeLocal = THREADS_PER_LOCAL_MERGE * ELEMS_PER_THREAD_LOCAL_MERGE;

    // Number of phases, which can be executed in shared memory (stride is lower than shared memory size)
    uint_t phasesBitonicSort = log2((double)min(tableLenPower2, elemsPerBlockBitonicSort));
    uint_t phasesMergeLocal = log2((double)min(tableLenPower2, elemsPerBlockMergeLocal));
    uint_t phasesAll = log2((double)tableLenPower2);

    // Sorts blocks of input data with bitonic sort
    runBitoicSortKernel();

    // Bitonic merge
    for (uint_t phase = phasesBitonicSort + 1; phase <= phasesAll; phase++)
    {
        uint_t step = phase;
        while (step > phasesMergeLocal)
        {
            runBitonicMergeGlobalKernel(phase, step);
            step--;
        }

        runBitoicMergeLocalKernel(phase, step);
    }
}
