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
Sorts sub-blocks of data with merge sort.
*/
void runMergeSortKernel(data_t *dataTable, uint_t tableLen, order_t sortOrder) {
    // Every thread loads and sorts 2 elements
    uint_t threadBlockSize = SHARED_MEM_SIZE / 2;
    uint_t sharedMemSize = SHARED_MEM_SIZE;

    dim3 dimGrid((tableLen - 1) / (threadBlockSize * 2) + 1, 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

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
    uint_t numAllSamples = tableLen / SUB_BLOCK_SIZE;
    uint_t threadBlockSize = min(numAllSamples, SHARED_MEM_SIZE);
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
    uint_t numAllSamples = tableLen / SUB_BLOCK_SIZE;
    uint_t threadBlockSize = min(numAllSamples, SHARED_MEM_SIZE);

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
    uint_t subBlocksPerMergedBlock = sortedBlockSize / SUB_BLOCK_SIZE * 2;
    uint_t numMergedBlocks = tableLen / (sortedBlockSize * 2);

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

double sortParallel(
    data_t *h_output, data_t *d_dataTable, data_t *d_dataBuffer, sample_t *d_samples, uint_t *d_ranksEven,
    data_t *d_ranksOdd, uint_t tableLen, order_t sortOrder
)
{
    LARGE_INTEGER timer;
    cudaError_t error;

    startStopwatch(&timer);
    runMergeSortKernel(d_dataTable, tableLen, sortOrder);

    for (uint_t sortedBlockSize = SHARED_MEM_SIZE; sortedBlockSize < tableLen; sortedBlockSize *= 2)
    {
        data_t* temp = d_dataTable;
        d_dataTable = d_dataBuffer;
        d_dataBuffer = temp;

        runGenerateSamplesKernel(d_dataBuffer, d_samples, tableLen, sortedBlockSize, sortOrder);
        runGenerateRanksKernel(
            d_dataBuffer, d_samples, d_ranksEven, d_ranksOdd, tableLen, sortedBlockSize, sortOrder
        );
        runMergeKernel(
            d_dataBuffer, d_dataTable, d_ranksEven, d_ranksOdd, tableLen, sortedBlockSize, sortOrder
        );
    }

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    double time = endStopwatch(timer);

    error = cudaMemcpy(h_output, d_dataTable, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    return time;
}
