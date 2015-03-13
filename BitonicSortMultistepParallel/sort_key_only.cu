#include <stdio.h>
#include <math.h>
#include <Windows.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"
#include "../Utils/cuda.h"
#include "constants.h"
#include "kernels_key_only.h"
#include "sort.h"


/*
Sorts sub-blocks of input data with bitonic sort.
*/
template <order_t sortOrder>
void BitonicSortMultistepParallel::runBitoicSortKernel(data_t *d_keys, uint_t arrayLength)
{
    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT_KO * ELEMS_PER_THREAD_BITONIC_SORT_KO;
    uint_t sharedMemSize = elemsPerThreadBlock * sizeof(*d_keys);

    dim3 dimGrid((arrayLength - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_BITONIC_SORT_KO, 1, 1);

    bitonicSortKernel<sortOrder><<<dimGrid, dimBlock, sharedMemSize>>>(d_keys, arrayLength);
}

/*
Runs bitonic multistep merge kernel, which uses registers. Multistep means, that every thread reads
multiple elements and sorts them according to bitonic sort exchanges for N steps ahead.
*/
template <order_t sortOrder>
void BitonicSortMultistepParallel::runMultiStepKernel(
    data_t *d_keys, uint_t arrayLength, uint_t phase, uint_t step, uint_t degree
)
{
    // Breaks table len into its power of 2 length and the remainder.
    uint_t power2arrayLen = previousPowerOf2(arrayLength);
    uint_t residueArrayLen = arrayLength % power2arrayLen;

    uint_t partitionSize = (power2arrayLen - 1) / (1 << degree) + 1;
    // For remainder the size of partition has to be calculated explicitly, becaause it depends on
    // remainder size, step and degree
    if (residueArrayLen > 0)
    {
        // The size of one sub-block which is sorted with same group of comparissons.
        uint_t subBlockSize = 1 << step;
        // Rouns the residue size to the next power of sub-block size
        uint_t power2residueArrayLen = roundUp(residueArrayLen, subBlockSize);
        partitionSize += min(residueArrayLen, (power2residueArrayLen - 1) / (1 << degree) + 1);
    }

    uint_t threadBlockSize = min(partitionSize, THREADS_PER_MULTISTEP_MERGE_KO);
    dim3 dimGrid((partitionSize - 1) / threadBlockSize + 1, 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    bool isFirstStepOfPhase = phase == step;

    if (degree == 1)
    {
        multiStep1Kernel<sortOrder><<<dimGrid, dimBlock>>>(d_keys, arrayLength, step);
    }
    else if (degree == 2)
    {
        multiStep2Kernel<sortOrder><<<dimGrid, dimBlock>>>(d_keys, arrayLength, step);
    }
    else if (degree == 3)
    {
        multiStep3Kernel<sortOrder><<<dimGrid, dimBlock>>>(d_keys, arrayLength, step);
    }
    else if (degree == 4)
    {
        multiStep4Kernel<sortOrder><<<dimGrid, dimBlock>>>(d_keys, arrayLength, step);
    }
    else if (degree == 5)
    {
        multiStep5Kernel<sortOrder><<<dimGrid, dimBlock>>>(d_keys, arrayLength, step);
    }
    else if (degree == 6)
    {
        multiStep6Kernel<sortOrder><<<dimGrid, dimBlock>>>(d_keys, arrayLength, step);
    }
}

/*
Merges array, if data blocks are larger than shared memory size. Needed because of normalized bitonic sort,
which uses different access pattern for first step of every phase than in other steps.
*/
template <order_t sortOrder>
void BitonicSortMultistepParallel::runBitonicMergeGlobalKernel(data_t *d_keys, uint_t arrayLength, uint_t phase)
{
    uint_t elemsPerThreadBlock = THREADS_PER_GLOBAL_MERGE_KO * ELEMS_PER_THREAD_GLOBAL_MERGE_KO;
    dim3 dimGrid((arrayLength - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_GLOBAL_MERGE_KO, 1, 1);

    bitonicMergeGlobalKernel<sortOrder><<<dimGrid, dimBlock>>>(d_keys, arrayLength, phase);
}

/*
Merges array when stride is lower than shared memory size. It executes all remaining STEPS of current PHASE.
*/
template <order_t sortOrder>
void BitonicSortMultistepParallel::runBitoicMergeLocalKernel(
    data_t *d_keys, uint_t arrayLength, uint_t phase, uint_t step
)
{
    // Every thread loads and sorts 2 elements
    uint_t elemsPerThreadBlock = THREADS_PER_LOCAL_MERGE_KO * ELEMS_PER_THREAD_LOCAL_MERGE_KO;
    uint_t sharedMemSize = elemsPerThreadBlock * sizeof(*d_keys);
    dim3 dimGrid((arrayLength - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_LOCAL_MERGE_KO, 1, 1);

    bool isFirstStepOfPhase = phase == step;

    if (isFirstStepOfPhase)
    {
        bitonicMergeLocalKernel<sortOrder, true><<<dimGrid, dimBlock, sharedMemSize>>>(d_keys, arrayLength, step);
    }
    else
    {
        bitonicMergeLocalKernel<sortOrder, false><<<dimGrid, dimBlock, sharedMemSize>>>(d_keys, arrayLength, step);
    }
}

/*
Sorts data with NORMALIZED BITONIC SORT.
*/
template <order_t sortOrder>
void BitonicSortMultistepParallel::bitonicSortMultistepParallel(data_t *d_keys, uint_t arrayLength)
{
    uint_t arrayLengthPower2 = nextPowerOf2(arrayLength);
    uint_t elemsPerBlockBitonicSort = THREADS_PER_BITONIC_SORT_KO * ELEMS_PER_THREAD_BITONIC_SORT_KO;
    uint_t elemsPerBlockMergeLocal = THREADS_PER_LOCAL_MERGE_KO * ELEMS_PER_THREAD_LOCAL_MERGE_KO;

    // Number of phases, which can be executed in shared memory (stride is lower than shared memory size)
    uint_t phasesBitonicSort = log2((double)min(arrayLengthPower2, elemsPerBlockBitonicSort));
    uint_t phasesMergeLocal = log2((double)min(arrayLengthPower2, elemsPerBlockMergeLocal));
    uint_t phasesAll = log2((double)arrayLengthPower2);

    runBitoicSortKernel<sortOrder>(d_keys, arrayLength);

    // Bitonic merge
    for (uint_t phase = phasesBitonicSort + 1; phase <= phasesAll; phase++)
    {
        uint_t step = phase;

        if (step > phasesMergeLocal)
        {
            // Global NORMALIZED bitonic merge for first step of phase, where different pattern of exchanges
            // is used compared to other steps
            runBitonicMergeGlobalKernel<sortOrder>(d_keys, arrayLength, phase);
            step--;

            // Multisteps
            for (uint_t degree = min(MAX_MULTI_STEP_KO, step - phasesMergeLocal); degree > 0; degree--)
            {
                for (; step >= phasesMergeLocal + degree; step -= degree)
                {
                    runMultiStepKernel<sortOrder>(d_keys, arrayLength, phase, step, degree);
                }
            }
        }

        runBitoicMergeLocalKernel<sortOrder>(d_keys, arrayLength, phase, step);
    }
}

/*
Wrapper for bitonic sort method.
The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
*/
void BitonicSortMultistepParallel::sortKeyOnly()
{
    if (_sortOrder == ORDER_ASC)
    {
        bitonicSortMultistepParallel<ORDER_ASC>(_d_keys, _arrayLength);
    }
    else
    {
        bitonicSortMultistepParallel<ORDER_DESC>(_d_keys, _arrayLength);
    }
}