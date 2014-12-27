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
#include "kernels.h"


/*
Returns the size of array that needs to be merged. If array size is power of 2, than array size is returned.
In opposite case array size is broken into 2 parts:
- main part (previous power of 2 of table length)
- remainder (table length - main part size)

> Remainder needs to be merged only until it is sorted (function returns size of whole array rounded up).
> After that only main part of the array has to be merged (function returns size of "main part").
> In last merge phase "main part" and "remainder" have to be merged.
*/
uint_t calculateMergeTableSize(uint_t tableLen, uint_t sortedBlockSize)
{
    uint_t tableLenMerge = previousPowerOf2(tableLen);
    uint_t mergedBlockSize = 2 * sortedBlockSize;

    // Table length is already a power of 2
    if (tableLenMerge != tableLen)
    {
        // Number of elements over the power of 2 length
        uint_t remainder = tableLen - tableLenMerge;

        // Remainder needs to be merged only if it is GREATER or EQUAL than current sorted block size. If it is
        // SMALLER than current sorted block size, this means that it has already been sorted. In that case only
        // the main part of array (previous power of 2) has to be merged.
        if (remainder >= sortedBlockSize)
        {
            tableLenMerge += roundUp(remainder, 2 * sortedBlockSize);
        }
        // In last merge phase the whole array has to be merged (main part of array + remainder)
        else if (tableLenMerge == sortedBlockSize)
        {
            tableLenMerge += sortedBlockSize;
        }
    }

    return tableLenMerge;
}

/*
Adds padding of MAX/MIN values to input table, deppending if sort order is ascending or descending. This is
needed, if table length is not power of 2. In order for bitonic sort to work, table length has to be power of 2.
*/
void runAddPaddingKernel(data_t *dataTable, data_t *dataBuffer, uint_t tableLen, order_t sortOrder)
{
    if (isPowerOfTwo(tableLen))
    {
        return;
    }

    uint_t sortedBlockSize = THREADS_PER_MERGE_SORT * ELEMS_PER_THREAD_MERGE_SORT;
    uint_t paddingLength = 0;

    // Table size is rounded to the next power of 2
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
        addPaddingKernel<MAX_VAL><<<dimGrid, dimBlock>>>(dataTable, dataBuffer, tableLen, paddingLength);
    }
    else
    {
        addPaddingKernel<MIN_VAL><<<dimGrid, dimBlock>>>(dataTable, dataBuffer, tableLen, paddingLength);
    }
}

/*
Sorts sub-blocks of data with merge sort.
*/
void runMergeSortKernel(data_t *dataTable, uint_t tableLen, order_t sortOrder)
{
    // For arrays shorther than THREADS_PER_MERGE_SORT * ELEMS_PER_THREAD_MERGE_SORT
    uint_t elemsPerThreadBlock = min(tableLen, THREADS_PER_MERGE_SORT * ELEMS_PER_THREAD_MERGE_SORT);
    // In case table length is not power of 2, table is padded with MIN/MAX values to the next power of 2.
    // Padded MIN/MAX elements don't need to be sorted (they are already "sorted"), that's why array is
    // sorted only to the next multiply of number of elements processed by one thread block.
    uint_t tableLenRoundedUp = roundUp(tableLen, elemsPerThreadBlock);

    // "2 *" because buffer shared memory is used in kernel alongside primary shared memory
    uint_t sharedMemSize = 2 * elemsPerThreadBlock * sizeof(*dataTable);

    dim3 dimGrid((tableLenRoundedUp - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(min(tableLenRoundedUp / ELEMS_PER_THREAD_MERGE_SORT, THREADS_PER_MERGE_SORT), 1, 1);

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
If array length is not power of 2, then array is split into 2 parts:
- main part (previous power of 2 of table length)
- remainder (table length - main part size)

Remainder needs to be merged only until it is sorted. After that only main part of the array has to be merged.
For every merge phase data is passed from main array to buffer (and vice versa)
In last merge phase "main part" and "remainder" have to be merged. If number of merge phases is odd, than
"remainder" is located in buffer, while "main part" is located in primary array. In that case "remainder" has
to be coppied to "main array", so they can be merged.
*/
void copyPaddedElements(
    data_t *toArray, data_t *fromArray, uint_t tableLen, uint_t sortedBlockSize, uint_t &lastPaddingMergePhase
)
{
    uint_t tableLenMerge = previousPowerOf2(tableLen);
    uint_t remainder = tableLen - tableLenMerge;

    // If remainder has to be merged || if this is last merge phase (main part and remainder have to be merged)
    if (remainder >= sortedBlockSize || tableLenMerge == sortedBlockSize)
    {
        // Calculates current merge phase
        uint_t currentMergePhase = log2((double)(2 * sortedBlockSize));
        uint_t phaseDifference = currentMergePhase - lastPaddingMergePhase;

        // If difference between phase when remainder was last merged and current phase is EVEN, this means
        // that remainder is located in buffer while main part is located in primary array. In that case
        // remainder is coppied into primary array.
        if (phaseDifference % 2 == 0)
        {
            cudaError_t error = cudaMemcpy(
                toArray, fromArray, remainder * sizeof(*toArray), cudaMemcpyDeviceToDevice
            );
            checkCudaError(error);
        }

        // Saves last phase when remainder was merged.
        lastPaddingMergePhase = currentMergePhase;
    }
}

/*
Generates array of ranks/boundaries of sub-block, which will be merged.
*/
void runGenerateRanksKernel(
    data_t *dataTable, uint_t *ranksEven, uint_t *ranksOdd, uint_t tableLen, uint_t sortedBlockSize,
    order_t sortOrder
)
{
    uint_t tableLenRoundedUp = calculateMergeTableSize(tableLen, sortedBlockSize);
    uint_t numAllSamples = (tableLenRoundedUp - 1) / SUB_BLOCK_SIZE + 1;
    uint_t threadBlockSize = min(numAllSamples, THREADS_PER_GEN_RANKS);

    dim3 dimGrid((numAllSamples - 1) / threadBlockSize + 1, 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    if (sortOrder == ORDER_ASC)
    {
        generateRanksKernel<ORDER_ASC><<<dimGrid, dimBlock>>>(
            dataTable, ranksEven, ranksOdd, sortedBlockSize
        );
    }
    else
    {
        generateRanksKernel<ORDER_DESC><<<dimGrid, dimBlock>>>(
            dataTable, ranksEven, ranksOdd, sortedBlockSize
        );
    }
}

/*
Executes merge kernel, which merges all consecutive sorted blocks.
*/
void runMergeKernel(
    data_t *input, data_t *output, uint_t *ranksEven, uint_t *ranksOdd, uint_t tableLen, uint_t sortedBlockSize,
    order_t sortOrder
)
{
    uint_t tableLenMerge = calculateMergeTableSize(tableLen, sortedBlockSize);
    uint_t mergedBlockSize = 2 * sortedBlockSize;

    // Sub-blocks of size SUB_BLOCK_SIZE per one merged block
    uint_t subBlocksPerMergedBlock = (mergedBlockSize - 1) / SUB_BLOCK_SIZE + 1;
    // Number of merged blocks
    uint_t numMergedBlocks = (tableLenMerge - 1) / mergedBlockSize + 1;

    // "+ 1" is used because 1 thread block more is needed than number of samples/ranks/splitters per
    // merged block (eg: if we cut something n times, we get n + 1 pieces)
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
    data_t *h_output, data_t *d_dataTable, data_t *d_dataBuffer, uint_t *d_ranksEven, data_t *d_ranksOdd,
    uint_t tableLen, order_t sortOrder
)
{
    LARGE_INTEGER timer;
    cudaError_t error;

    startStopwatch(&timer);

    runAddPaddingKernel(d_dataTable, d_dataBuffer, tableLen, sortOrder);
    runMergeSortKernel(d_dataTable, tableLen, sortOrder);

    uint_t sortedBlockSize = THREADS_PER_MERGE_SORT * ELEMS_PER_THREAD_MERGE_SORT;
    // Last merge phase when "remainder" was merged (part of array over it's last power of 2 in length)
    uint_t lastPaddingMergePhase = log2((double)(sortedBlockSize));
    uint_t tableLenPrevPower2 = previousPowerOf2(tableLen);

    while (sortedBlockSize < tableLen)
    {
        data_t* temp = d_dataTable;
        d_dataTable = d_dataBuffer;
        d_dataBuffer = temp;

        copyPaddedElements(
            d_dataBuffer + tableLenPrevPower2, d_dataTable + tableLenPrevPower2, tableLen, sortedBlockSize,
            lastPaddingMergePhase
        );
        runGenerateRanksKernel(d_dataBuffer, d_ranksEven, d_ranksOdd, tableLen, sortedBlockSize, sortOrder);
        runMergeKernel(
            d_dataBuffer, d_dataTable, d_ranksEven, d_ranksOdd, tableLen, sortedBlockSize, sortOrder
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
