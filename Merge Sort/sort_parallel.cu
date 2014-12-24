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
Sorts sub-blocks of data with merge sort.
*/
void runMergeSortKernel(data_t *dataTable, uint_t tableLen, order_t sortOrder) {
    cudaError_t error;
    LARGE_INTEGER timer;

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

///*
//Generates array of samples used to partition the table for merge step.
//*/
//void runGenerateSamplesKernel(el_t *table, el_t *samples, uint_t tableLen, uint_t sortedBlockSize,
//                              bool orderAsc) {
//    cudaError_t error;
//    LARGE_INTEGER timer;
//
//    uint_t numAllSamples = tableLen / SUB_BLOCK_SIZE;
//    uint_t threadBlockSize = min(numAllSamples, SHARED_MEM_SIZE);
//    dim3 dimGrid((numAllSamples - 1) / threadBlockSize + 1, 1, 1);
//    dim3 dimBlock(threadBlockSize, 1, 1);
//
//    startStopwatch(&timer);
//    generateSamplesKernel<<<dimGrid, dimBlock>>>(table, samples, sortedBlockSize, orderAsc);
//    /*error = cudaDeviceSynchronize();
//    checkCudaError(error);
//    endStopwatch(timer, "Executing kernel for generating samples");*/
//}
//
///*
//Generates ranks/limits of sub-blocks that need to be merged.
//*/
//void runGenerateRanksKernel(el_t *table, el_t *samples, uint_t *ranksEven, uint_t *ranksOdd,
//                            uint_t tableLen, uint_t sortedBlockSize, bool orderAsc) {
//    cudaError_t error;
//    LARGE_INTEGER timer;
//
//    uint_t numAllSamples = tableLen / SUB_BLOCK_SIZE;
//    uint_t threadBlockSize = min(numAllSamples, SHARED_MEM_SIZE);
//    dim3 dimGrid((numAllSamples - 1) / threadBlockSize + 1, 1, 1);
//    dim3 dimBlock(threadBlockSize, 1, 1);
//
//    startStopwatch(&timer);
//    generateRanksKernel<<<dimGrid, dimBlock>>>(
//        table, samples, ranksEven, ranksOdd, sortedBlockSize, orderAsc
//    );
//    /*error = cudaDeviceSynchronize();
//    checkCudaError(error);
//    endStopwatch(timer, "Executing generate ranks kernel");*/
//}
//
///*
//Executes merge kernel, which merges all consecutive sorted blocks in data.
//*/
//void runMergeKernel(el_t *input, el_t *output, uint_t *ranksEven, uint_t *ranksOdd, uint_t tableLen,
//                    uint_t sortedBlockSize) {
//    cudaError_t error;
//    LARGE_INTEGER timer;
//
//    uint_t subBlocksPerMergedBlock = sortedBlockSize / SUB_BLOCK_SIZE * 2;
//    uint_t numMergedBlocks = tableLen / (sortedBlockSize * 2);
//    dim3 dimGrid(subBlocksPerMergedBlock + 1, numMergedBlocks, 1);
//    dim3 dimBlock(SUB_BLOCK_SIZE, 1, 1);
//
//    startStopwatch(&timer);
//    mergeKernel<<<dimGrid, dimBlock>>>(
//        input, output, ranksEven, ranksOdd, tableLen, sortedBlockSize
//    );
//    /*error = cudaDeviceSynchronize();
//    checkCudaError(error);
//    endStopwatch(timer, "Executing merge kernel");*/
//}

double sortParallel(
    data_t *h_output, data_t *d_dataTable, data_t *dataBuffer, uint_t tableLen, order_t sortOrder
)
{
    data_t *d_input, *d_output, *d_buffer;
    data_t *d_samples;
    uint_t *d_ranksEven, *d_ranksOdd;
    uint_t samplesLen = tableLen / SUB_BLOCK_SIZE;

    LARGE_INTEGER timer;
    cudaError_t error;

    startStopwatch(&timer);
    runMergeSortKernel(d_dataTable, tableLen, sortOrder);

    //for (uint_t sortedBlockSize = SHARED_MEM_SIZE; sortedBlockSize < tableLen; sortedBlockSize *= 2) {
    //    el_t* temp = d_output;
    //    d_output = d_buffer;
    //    d_buffer = temp;

    //    runGenerateSamplesKernel(d_buffer, d_samples, tableLen, sortedBlockSize, orderAsc);
    //    runGenerateRanksKernel(d_buffer, d_samples, d_ranksEven, d_ranksOdd, tableLen,
    //                           sortedBlockSize, orderAsc);
    //    runMergeKernel(d_buffer, d_output, d_ranksEven, d_ranksOdd, tableLen, sortedBlockSize);
    //}

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    double time = endStopwatch(timer);

    error = cudaMemcpy(h_output, d_dataTable, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    return time;
}
