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
#include "kernels.h"


/*
Sorts sub-blocks of input data with bitonic sort.
*/
void runBitoicSortKernel(data_t *keys, data_t *values, uint_t tableLen, order_t sortOrder) {
    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    uint_t sharedMemSize = 2 * elemsPerThreadBlock * sizeof(*keys);  // "2 *" becaues of key-value pairs

    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_BITONIC_SORT, 1, 1);

    if (sortOrder == ORDER_ASC)
    {
        bitonicSortKernel<ORDER_ASC><<<dimGrid, dimBlock, sharedMemSize>>>(keys, values, tableLen);
    }
    else
    {
        bitonicSortKernel<ORDER_DESC><<<dimGrid, dimBlock, sharedMemSize>>>(keys, values, tableLen);
    }
}

/*
Runs bitonic multistep merge kernel, which uses registers. Multistep means, that every thread reads
multiple elements and sorts them according to bitonic sort exchanges for N steps ahead.
*/
void runMultiStepKernel(
    data_t *table, uint_t tableLen, uint_t phase, uint_t step, uint_t degree, order_t sortOrder
)
{
    // Breaks table len into its power of 2 length and the remainder.
    uint_t power2tableLen = previousPowerOf2(tableLen);
    uint_t residueTableLen = tableLen % power2tableLen;

    uint_t partitionSize = (power2tableLen - 1) / (1 << degree) + 1;
    // For remainder the size of partition has to be calculated explicitly, becaause it depends on
    // remainder size, step and degree
    if (residueTableLen > 0)
    {
        // The size of one sub-block which is sorted with same group of comparissons.
        uint_t subBlockSize = 1 << step;
        // Rouns the residue size to the next power of sub-block size
        uint_t power2residueTableLen = roundUp(residueTableLen, subBlockSize);
        partitionSize += min(residueTableLen, (power2residueTableLen - 1) / (1 << degree) + 1);
    }

    uint_t threadBlockSize = min(partitionSize, THREADS_PER_MULTISTEP_MERGE);
    dim3 dimGrid((partitionSize - 1) / threadBlockSize + 1, 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    bool isFirstStepOfPhase = phase == step;

    if (degree == 1)
    {
        if (sortOrder == ORDER_ASC)
        {
            multiStep1Kernel<ORDER_ASC><<<dimGrid, dimBlock>>>(table, tableLen, step);
        }
        else
        {
            multiStep1Kernel<ORDER_DESC><<<dimGrid, dimBlock>>>(table, tableLen, step);
        }
    }
    else if (degree == 2)
    {
        if (sortOrder == ORDER_ASC)
        {
            multiStep2Kernel<ORDER_ASC><<<dimGrid, dimBlock>>>(table, tableLen, step);
        }
        else
        {
            multiStep2Kernel<ORDER_DESC><<<dimGrid, dimBlock>>>(table, tableLen, step);
        }
    }
    else if (degree == 3)
    {
        if (sortOrder == ORDER_ASC)
        {
            multiStep3Kernel<ORDER_ASC><<<dimGrid, dimBlock>>>(table, tableLen, step);
        }
        else
        {
            multiStep3Kernel<ORDER_DESC><<<dimGrid, dimBlock>>>(table, tableLen, step);
        }
    }
    else if (degree == 4)
    {
        if (sortOrder == ORDER_ASC)
        {
            multiStep4Kernel<ORDER_ASC><<<dimGrid, dimBlock>>>(table, tableLen, step);
        }
        else
        {
            multiStep4Kernel<ORDER_DESC><<<dimGrid, dimBlock>>>(table, tableLen, step);
        }
    }
    else if (degree == 5)
    {
        if (sortOrder == ORDER_ASC)
        {
            multiStep5Kernel<ORDER_ASC><<<dimGrid, dimBlock>>>(table, tableLen, step);
        }
        else
        {
            multiStep5Kernel<ORDER_DESC><<<dimGrid, dimBlock>>>(table, tableLen, step);
        }
    }
    else if (degree == 6)
    {
        if (sortOrder == ORDER_ASC)
        {
            multiStep6Kernel<ORDER_ASC><<<dimGrid, dimBlock>>>(table, tableLen, step);
        }
        else
        {
            multiStep6Kernel<ORDER_DESC><<<dimGrid, dimBlock>>>(table, tableLen, step);
        }
    }
}

/*
Merges array, if data blocks are larger than shared memory size. It executes only of STEP on PHASE per
kernel launch.
*/
void runBitonicMergeGlobalKernel(
    data_t *keys, data_t *values, uint_t tableLen, uint_t phase, uint_t step, order_t sortOrder
)
{
    uint_t elemsPerThreadBlock = THREADS_PER_GLOBAL_MERGE * ELEMS_PER_THREAD_GLOBAL_MERGE;
    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_GLOBAL_MERGE, 1, 1);

    if (sortOrder == ORDER_ASC)
    {
        bitonicMergeGlobalKernel<ORDER_ASC><<<dimGrid, dimBlock>>>(keys, values, tableLen, step);
    }
    else
    {
        bitonicMergeGlobalKernel<ORDER_DESC><<<dimGrid, dimBlock>>>(keys, values, tableLen, step);
    }
}

/*
Merges array when stride is lower than shared memory size. It executes all remaining STEPS of current PHASE.
*/
void runBitoicMergeLocalKernel(data_t *keys, data_t *values, uint_t tableLen, uint_t phase, uint_t step, order_t sortOrder)
{
    // Every thread loads and sorts 2 elements
    uint_t elemsPerThreadBlock = THREADS_PER_LOCAL_MERGE * ELEMS_PER_THREAD_LOCAL_MERGE;
    uint_t sharedMemSize = 2 * elemsPerThreadBlock * sizeof(*keys);  // "2 *" becaues of key-value pairs
    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_LOCAL_MERGE, 1, 1);

    bool isFirstStepOfPhase = phase == step;

    if (sortOrder == ORDER_ASC)
    {
        if (isFirstStepOfPhase) {
            bitonicMergeLocalKernel<ORDER_ASC, true><<<dimGrid, dimBlock, sharedMemSize>>>(
                keys, values, tableLen, step
            );
        }
        else
        {
            bitonicMergeLocalKernel<ORDER_ASC, false><<<dimGrid, dimBlock, sharedMemSize>>>(
                keys, values, tableLen, step
            );
        }
    }
    else
    {
        if (isFirstStepOfPhase) {
            bitonicMergeLocalKernel<ORDER_DESC, true><<<dimGrid, dimBlock, sharedMemSize>>>(
                keys, values, tableLen, step
            );
        }
        else
        {
            bitonicMergeLocalKernel<ORDER_DESC, false><<<dimGrid, dimBlock, sharedMemSize>>>(
                keys, values, tableLen, step
            );
        }
    }
}

/*
Sorts data with NORMALIZED BITONIC SORT.
*/
double sortParallel(
    data_t *h_keys, data_t *h_values, data_t *d_keys, data_t *d_values, uint_t tableLen, order_t sortOrder
)
{
    uint_t tableLenPower2 = nextPowerOf2(tableLen);
    uint_t elemsPerBlockBitonicSort = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    uint_t elemsPerBlockMergeLocal = THREADS_PER_LOCAL_MERGE * ELEMS_PER_THREAD_LOCAL_MERGE;

    // Number of phases, which can be executed in shared memory (stride is lower than shared memory size)
    uint_t phasesBitonicSort = log2((double)min(tableLenPower2, elemsPerBlockBitonicSort));
    uint_t phasesMergeLocal = log2((double)min(tableLenPower2, elemsPerBlockMergeLocal));
    uint_t phasesAll = log2((double)tableLenPower2);

    LARGE_INTEGER timer;
    cudaError_t error;

    startStopwatch(&timer);
    runBitoicSortKernel(d_keys, d_values, tableLen, sortOrder);

    // Bitonic merge
    for (uint_t phase = phasesBitonicSort + 1; phase <= phasesAll; phase++)
    {
        uint_t step = phase;

        if (step > phasesMergeLocal && step == phase)
        {
            // Global NORMALIZED bitonic merge for first step of phase, where different pattern of exchanges
            // is used compared to other steps
            runBitonicMergeGlobalKernel(d_keys, d_values, tableLen, phase, step, sortOrder);
            step--;

            // Multisteps
            for (uint_t degree = min(MAX_MULTI_STEP, step - phasesMergeLocal); degree > 0; degree--)
            {
                for (; step >= phasesMergeLocal + degree; step -= degree)
                {
                    runMultiStepKernel(d_keys, tableLen, phase, step, degree, sortOrder);
                }
            }
        }

        runBitoicMergeLocalKernel(d_keys, d_values, tableLen, phase, step, sortOrder);
    }

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    double time = endStopwatch(timer);

    error = cudaMemcpy(h_keys, d_keys, tableLen * sizeof(*h_keys), cudaMemcpyDeviceToHost);
    checkCudaError(error);
    error = cudaMemcpy(h_values, d_values, tableLen * sizeof(*h_values), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    return time;
}
