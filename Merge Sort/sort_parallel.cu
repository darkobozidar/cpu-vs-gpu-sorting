#include <stdio.h>
#include <Windows.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_types.h"
#include "constants.h"
#include "utils_cuda.h"
#include "utils_host.h"
#include "kernels.h"


/*
Initializes memory needed for storing data being sorted.
*/
void memoryDataInit(el_t *h_input, el_t **d_input, el_t **d_output, el_t **d_buffer, uint_t tableLen) {
    cudaError_t error;

    error = cudaMalloc(d_input, tableLen * sizeof(**d_input));
    checkCudaError(error);
    error = cudaMalloc(d_output, tableLen * sizeof(**d_output));
    checkCudaError(error);
    error = cudaMalloc(d_buffer, tableLen * sizeof(**d_buffer));
    checkCudaError(error);

    error = cudaMemcpy(*d_input, h_input, tableLen * sizeof(**d_input), cudaMemcpyHostToDevice);
    checkCudaError(error);
}

/*
Initializes memory needed for generating sub-block limits.
*/
void memoryMergeInit(sample_t **samples, uint_t **ranksEven, uint_t **ranksOdd,
                     uint_t samplesLen, uint_t ranksLen) {
    cudaError_t error;

    error = cudaMalloc(samples, samplesLen * sizeof(**samples));
    checkCudaError(error);
    error = cudaMalloc(ranksEven, samplesLen * sizeof(**ranksEven));
    checkCudaError(error);
    error = cudaMalloc(ranksOdd, samplesLen * sizeof(**ranksOdd));
    checkCudaError(error);
}

/*
Sorts sub-blocks of data with merge sort.
*/
void runMergeSortKernel(el_t *input, el_t *output, uint_t tableLen, bool orderAsc) {
    cudaError_t error;
    LARGE_INTEGER timer;

    // Every thread loads and sorts 2 elements
    uint_t threadBlockSize = SHARED_MEM_SIZE / 2;
    dim3 dimGrid((tableLen - 1) / (threadBlockSize * 2) + 1, 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    startStopwatch(&timer);
    mergeSortKernel<<<dimGrid, dimBlock>>>(input, output, orderAsc);
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing Merge sort Kernel");*/
}

void runGenerateSamplesKernel(el_t *table, sample_t *samples, uint_t tableLen, uint_t sortedBlockSize) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t numAllSamples = tableLen / SUB_BLOCK_SIZE;
    uint_t threadBlockSize = min(numAllSamples, SHARED_MEM_SIZE);
    dim3 dimGrid((numAllSamples - 1) / threadBlockSize + 1, 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    startStopwatch(&timer);
    generateSamplesKernel<<<dimGrid, dimBlock>>>(table, samples, sortedBlockSize);
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing kernel for sample extraction");*/
}

/*
Generates ranks of sub-blocks that need to be merged.
*/
void runGenerateRanksKernel(el_t *table, sample_t *samples, uint_t *ranksEven, uint_t *ranksOdd,
                            uint_t tableLen, uint_t sortedBlockSize) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t numAllSamples = tableLen / SUB_BLOCK_SIZE;
    uint_t threadBlockSize = min(numAllSamples, SHARED_MEM_SIZE);
    dim3 dimGrid((numAllSamples - 1) / threadBlockSize + 1, 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    startStopwatch(&timer);
    generateRanksKernel<<<dimGrid, dimBlock>>>(
        table, samples, ranksEven, ranksOdd, tableLen, sortedBlockSize
    );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing Generate ranks kernel");*/
}

/*
Executes merge kernel, which merges all consecutive sorted blocks in data.
*/
void runMergeKernel(el_t *input, el_t *output, uint_t *ranksEven, uint_t *ranksOdd, uint_t tableLen,
                    uint_t sortedBlockSize) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t subBlocksPerMergedBlock = sortedBlockSize / SUB_BLOCK_SIZE * 2;
    uint_t numMergedBlocks = tableLen / (sortedBlockSize * 2);
    dim3 dimGrid(subBlocksPerMergedBlock + 1, numMergedBlocks, 1);
    dim3 dimBlock(SUB_BLOCK_SIZE, 1, 1);

    startStopwatch(&timer);
    mergeKernel << <dimGrid, dimBlock>> >(
        input, output, ranksEven, ranksOdd, tableLen, sortedBlockSize, SUB_BLOCK_SIZE
    );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing merge kernel");*/
}

void sortParallel(el_t *h_input, el_t *h_output, uint_t tableLen, bool orderAsc) {
    el_t *d_input, *d_output, *d_buffer;
    sample_t *d_samples;
    uint_t *d_ranksEven, *d_ranksOdd;
    uint_t samplesLen = tableLen / SUB_BLOCK_SIZE;
    uint_t ranksLen = samplesLen;

    LARGE_INTEGER timer;
    cudaError_t error;

    memoryDataInit(h_input, &d_input, &d_output, &d_buffer, tableLen);
    memoryMergeInit(&d_samples, &d_ranksEven, &d_ranksOdd, samplesLen, ranksLen);

    startStopwatch(&timer);
    runMergeSortKernel(d_input, d_output, tableLen, orderAsc);

    for (uint_t sortedBlockSize = SHARED_MEM_SIZE; sortedBlockSize < tableLen; sortedBlockSize *= 2) {
        el_t* temp = d_output;
        d_output = d_buffer;
        d_buffer = temp;

        runGenerateSamplesKernel(d_buffer, d_samples, tableLen, sortedBlockSize);
        runGenerateRanksKernel(d_buffer, d_samples, d_ranksEven, d_ranksOdd, tableLen, sortedBlockSize);
        runMergeKernel(d_buffer, d_output, d_ranksEven, d_ranksOdd, tableLen, sortedBlockSize);
    }
    error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing parallel merge sort.");

    error = cudaMemcpy(h_output, d_output, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);
}
