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
#include "kernels.h"


/*
Adds padding of MAX/MIN values to input table, deppending if sort order is ascending or descending. This is
needed, if table length is not power of 2. In order for bitonic sort to work, table length has to be power of 2.
*/
uint_t runAddPaddingKernel(data_t *dataTable, uint_t tableLen, order_t sortOrder)
{
    if (isPowerOfTwo(tableLen))
    {
        return tableLen;
    }

    uint_t sortedBlockSize = THREADS_PER_MERGE_SORT * ELEMS_PER_THREAD_MERGE_SORT;
    uint_t paddingLength = 0;

    // If table length is smaller than number of elements sorted by one thread block in merge sort kernel,
    // then table has to be padded to the next power of 2.
    if (tableLen < ELEMS_PER_THREAD_MERGE_SORT)
    {
        paddingLength = ELEMS_PER_THREAD_MERGE_SORT - tableLen;
    }
    else
    {
        paddingLength = nextPowerOf2(tableLen) - tableLen;
    }

    uint_t elemsPerThreadBlock = THREADS_PER_PADDING * ELEMS_PER_THREAD_PADDING;
    dim3 dimGrid((paddingLength - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_PADDING, 1, 1);

    // Depending on sort order different value is used for padding.
    if (sortOrder == ORDER_ASC)
    {
        addPaddingKernel<MAX_VAL><<<dimGrid, dimBlock>>>(dataTable, tableLen, paddingLength);
    }
    else
    {
        addPaddingKernel<MIN_VAL><<<dimGrid, dimBlock>>>(dataTable, tableLen, paddingLength);
    }

    return tableLen + paddingLength;
}

/*
Sorts sub-blocks of data with merge sort.
*/
void runMergeSortKernel(data_t *dataTable, uint_t tableLen, order_t sortOrder)
{
    uint_t elemsPerThreadBlock = min(tableLen, THREADS_PER_MERGE_SORT * ELEMS_PER_THREAD_MERGE_SORT);
    // "2 *" because buffer shared memory is used in kernel alongside primary shared memory
    uint_t sharedMemSize = 2 * elemsPerThreadBlock * sizeof(*dataTable);

    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(min(tableLen / ELEMS_PER_THREAD_MERGE_SORT, THREADS_PER_MERGE_SORT) , 1, 1);

    if (sortOrder == ORDER_ASC)
    {
        mergeSortKernel<ORDER_ASC><<<dimGrid, dimBlock, sharedMemSize>>>(dataTable);
    }
    else
    {
        mergeSortKernel<ORDER_DESC><<<dimGrid, dimBlock, sharedMemSize>>>(dataTable);
    }
}

/*
Generates array of samples used to partition the table for merge step.
*/
void runGenerateSamplesKernel(
    data_t *dataTable, sample_t *samples, uint_t tableLen, uint_t sortedBlockSize, order_t sortOrder
)
{
    uint_t numAllSamples = (tableLen - 1) / SUB_BLOCK_SIZE + 1;
    uint_t threadBlockSize = min(numAllSamples, THREADS_PER_GEN_SAMPLES);

    dim3 dimGrid((numAllSamples - 1) / threadBlockSize + 1, 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    if (sortOrder == ORDER_ASC)
    {
        generateSamplesKernel<ORDER_ASC><<<dimGrid, dimBlock>>>(dataTable, samples, sortedBlockSize);
    }
    else
    {
        generateSamplesKernel<ORDER_DESC><<<dimGrid, dimBlock>>>(dataTable, samples, sortedBlockSize);
    }
}

/*
Generates ranks/limits of sub-blocks that need to be merged.
*/
void runGenerateRanksKernel(
    data_t *dataTable, sample_t *samples, uint_t *ranksEven, uint_t *ranksOdd, uint_t tableLen,
    uint_t sortedBlockSize, order_t sortOrder
)
{
    uint_t numAllSamples = (tableLen - 1) / SUB_BLOCK_SIZE + 1;
    uint_t threadBlockSize = min(numAllSamples, THREADS_PER_GEN_RANKS);

    dim3 dimGrid((numAllSamples - 1) / threadBlockSize + 1, 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    if (sortOrder == ORDER_ASC)
    {
        generateRanksKernel<ORDER_ASC><<<dimGrid, dimBlock>>>(
            dataTable, samples, ranksEven, ranksOdd, sortedBlockSize
        );
    }
    else
    {
        generateRanksKernel<ORDER_DESC><<<dimGrid, dimBlock>>>(
            dataTable, samples, ranksEven, ranksOdd, sortedBlockSize
        );
    }
}

/*
Executes merge kernel, which merges all consecutive sorted blocks in data.
*/
void runMergeKernel(
    data_t *input, data_t *output, uint_t *ranksEven, uint_t *ranksOdd, uint_t tableLen, uint_t sortedBlockSize,
    order_t sortOrder
)
{
    uint_t subBlocksPerMergedBlock = (2 * sortedBlockSize - 1) / SUB_BLOCK_SIZE + 1;
    uint_t numMergedBlocks = (tableLen - 1) / (sortedBlockSize * 2) + 1;

    dim3 dimGrid(subBlocksPerMergedBlock + 1, numMergedBlocks, 1);
    dim3 dimBlock(SUB_BLOCK_SIZE, 1, 1);

    if (sortOrder == ORDER_ASC)
    {
        mergeKernel<ORDER_ASC><<<dimGrid, dimBlock>>>(
            input, output, ranksEven, ranksOdd, sortedBlockSize
        );
    }
    else
    {
        mergeKernel<ORDER_DESC><<<dimGrid, dimBlock>>>(
            input, output, ranksEven, ranksOdd, sortedBlockSize
        );
    }
}

/*
Sorts data with parallel merge sort.
*/
double sortParallel(
    data_t *h_output, data_t *d_dataTable, data_t *d_dataBuffer, sample_t *d_samples, uint_t *d_ranksEven,
    data_t *d_ranksOdd, uint_t tableLen, order_t sortOrder
)
{
    LARGE_INTEGER timer;
    cudaError_t error;

    startStopwatch(&timer);
    uint_t tableLenRoundedUp = runAddPaddingKernel(d_dataTable, tableLen, sortOrder);
    runMergeSortKernel(d_dataTable, tableLenRoundedUp, sortOrder);

    uint_t sortedBlockSize = THREADS_PER_MERGE_SORT * ELEMS_PER_THREAD_MERGE_SORT;

    while (sortedBlockSize < tableLen)
    {
        data_t* temp = d_dataTable;
        d_dataTable = d_dataBuffer;
        d_dataBuffer = temp;

        runGenerateSamplesKernel(d_dataBuffer, d_samples, tableLenRoundedUp, sortedBlockSize, sortOrder);
        runGenerateRanksKernel(
            d_dataBuffer, d_samples, d_ranksEven, d_ranksOdd, tableLenRoundedUp, sortedBlockSize, sortOrder
        );
        runMergeKernel(
            d_dataBuffer, d_dataTable, d_ranksEven, d_ranksOdd, tableLenRoundedUp, sortedBlockSize, sortOrder
        );

        sortedBlockSize *= 2;
    }

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    double time = endStopwatch(timer);

    error = cudaMemcpy(h_output, d_dataTable, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    return time;
}
