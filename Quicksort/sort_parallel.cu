#include <stdio.h>
#include <climits>
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
Initializes HOST memory needed for paralel sort implementation.
*/
void memoryInitHost(h_glob_seq_t **h_globalSeqHost, h_glob_seq_t **h_globalSeqHostBuffer,
                    d_glob_seq_t **h_globalSeqDev, uint_t **h_globalSeqIndexes, loc_seq_t **h_localSeq,
                    uint_t maxSequences, uint_t maxNumThreadBlocks) {
    *h_globalSeqHost = new h_glob_seq_t[maxSequences];
    *h_globalSeqHostBuffer = new h_glob_seq_t[maxSequences];
    *h_globalSeqDev = new d_glob_seq_t[maxSequences];
    *h_globalSeqIndexes = new uint_t[maxNumThreadBlocks];
    *h_localSeq = new loc_seq_t[maxSequences];
}

/*
Initializes DEVICE memory needed for paralel sort implementation.
*/
void memoryInitDevice(el_t *h_input, el_t **d_dataInput, el_t **d_dataBuffer, d_glob_seq_t **d_globalSeqDev,
                      uint_t **d_globalSeqIndexes, loc_seq_t **h_localSeq, uint_t tableLen,
                      uint_t maxSequences, uint_t maxNumThreadBlocks) {
    cudaError_t error;

    // Data memory allocation
    error = cudaMalloc(d_dataInput, tableLen * sizeof(**d_dataInput));
    checkCudaError(error);
    error = cudaMalloc(d_dataBuffer, tableLen * sizeof(**d_dataBuffer));
    checkCudaError(error);
    // Sequence metadata memory allocation
    error = cudaMalloc(d_globalSeqDev, maxSequences * sizeof(**d_globalSeqDev));
    checkCudaError(error);
    error = cudaMalloc(d_globalSeqIndexes, maxNumThreadBlocks * sizeof(**d_globalSeqIndexes));
    checkCudaError(error);
    error = cudaMalloc(h_localSeq, maxSequences * sizeof(**h_localSeq));
    checkCudaError(error);

    error = cudaMemcpy(*d_dataInput, h_input, tableLen * sizeof(**d_dataInput), cudaMemcpyHostToDevice);
    checkCudaError(error);
}

void runQuickSortGlobalKernel(el_t *input, el_t* output, d_glob_seq_t *h_devGlobalParams,
                              d_glob_seq_t *d_devGlobalParams, uint_t *h_globalSeqIndexes, uint_t *d_globalSeqIndexes,
                              uint_t hostWorkCounter, uint_t threadBlockCounter, uint_t tableLen) {
    cudaError_t error;
    LARGE_INTEGER timer;

    startStopwatch(&timer);

    error = cudaMemcpy(d_devGlobalParams, h_devGlobalParams, hostWorkCounter * sizeof(*d_devGlobalParams),
                       cudaMemcpyHostToDevice);
    checkCudaError(error);
    error = cudaMemcpy(d_globalSeqIndexes, h_globalSeqIndexes, threadBlockCounter * sizeof(*d_globalSeqIndexes),
                       cudaMemcpyHostToDevice);
    checkCudaError(error);

    // TODO comment shared memory size, 2 * size should be enough, because scan and min/max can be
    // performed in the same array
    quickSortGlobalKernel<<<threadBlockCounter, THREADS_PER_SORT_GLOBAL, 2 * THREADS_PER_SORT_GLOBAL>>>(
        input, output, d_devGlobalParams, d_globalSeqIndexes, tableLen
    );

    error = cudaMemcpy(h_devGlobalParams, d_devGlobalParams, hostWorkCounter * sizeof(*h_devGlobalParams),
                       cudaMemcpyDeviceToHost);
    checkCudaError(error);

    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing global parallel quicksort.");*/
}

void runQuickSortLocalKernel(el_t *input, el_t *output, loc_seq_t *h_localParams, loc_seq_t *d_localParams,
                             uint_t tableLen, uint_t numThreadBlocks, bool orderAsc) {
    cudaError_t error;
    LARGE_INTEGER timer;

    // The same shared memory array is used for counting elements greater/lower than pivot and for bitonic sort.
    // max(intra-block-scan array size, array size for bitonic sort)
    uint_t sharedMemSize = max(
        2 * THREADS_PER_SORT_LOCAL * sizeof(uint_t), BITONIC_SORT_SIZE_LOCAL * sizeof(*input)
    );
    dim3 dimGrid(numThreadBlocks, 1, 1);
    dim3 dimBlock(THREADS_PER_SORT_LOCAL, 1, 1);

    startStopwatch(&timer);
    error = cudaMemcpy(d_localParams, h_localParams, numThreadBlocks * sizeof(*d_localParams),
                       cudaMemcpyHostToDevice);
    checkCudaError(error);

    quickSortLocalKernel<<<dimGrid, dimBlock, sharedMemSize>>>(
        input, output, d_localParams, tableLen, orderAsc
    );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing local parallel quicksort.");*/
}

void runPrintTableKernel(el_t *table, uint_t tableLen) {
    printTableKernel<<<1, 1>>>(table, tableLen);
    cudaError_t error = cudaDeviceSynchronize();
    checkCudaError(error);
}

// TODO handle empty sub-blocks
void quickSort(el_t *hostData, el_t *dataInput, el_t *dataBuffer, h_glob_seq_t *h_hostGlobalParams,
               h_glob_seq_t *h_hostGlobalBuffer, d_glob_seq_t *h_devGlobalParams, d_glob_seq_t *d_devGlobalParams,
               uint_t *h_globalSeqIndexes, uint_t *d_globalSeqIndexes, loc_seq_t *h_localParams,
               loc_seq_t *d_localParams, uint_t tableLen, bool orderAsc) {
    // Set starting work
    uint_t minVal = min(min(hostData[0].key, hostData[tableLen / 2].key), hostData[tableLen - 1].key);
    uint_t maxVal = max(max(hostData[0].key, hostData[tableLen / 2].key), hostData[tableLen - 1].key);
    // TODO pass pivot to constructor
    h_hostGlobalParams[0].setInitSeq(tableLen, (minVal + maxVal) / 2);

    // Size of workstack
    uint_t workTotal = 1;
    uint_t hostWorkCounter = 1;
    uint_t localWorkCounter = 0;
    uint_t elemsPerThreadBlock = THREADS_PER_SORT_GLOBAL * ELEMENTS_PER_THREAD_GLOBAL;
    // Maximum number of sequences, which can be generated with global quicksort
    uint_t maxSequences = (tableLen - 1) / (MIN_PARTITION_SIZE_GLOBAL * 1) + 1;  // TODO replace 1 with constant
    cudaError_t error;

    // TODO if statement for initial sequence length
    while (workTotal < maxSequences) {
        uint_t threadBlockCounter = 0;

        // Store work to device
        for (uint_t workIdx = 0; workIdx < hostWorkCounter; workIdx++) {
            uint_t threadBlocksPerSequence = (h_hostGlobalParams[workIdx].length - 1) / elemsPerThreadBlock + 1;

            // For every thread block marks, which sequence they have to partiton (which work they have to perform)
            for (uint_t blockIdx = 0; blockIdx < threadBlocksPerSequence; blockIdx++) {
                h_globalSeqIndexes[threadBlockCounter++] = workIdx;
            }

            // Store work, that thread blocks assigned to current sequence have to perform
            h_devGlobalParams[workIdx].setFromHostSeq(h_hostGlobalParams[workIdx], threadBlocksPerSequence);
        }

        runQuickSortGlobalKernel(
            dataInput, dataBuffer, h_devGlobalParams, d_devGlobalParams, h_globalSeqIndexes,
            d_globalSeqIndexes, hostWorkCounter, threadBlockCounter, tableLen
        );

        runPrintTableKernel(dataBuffer, tableLen);

        uint_t oldHostWorkCounter = hostWorkCounter;
        hostWorkCounter = 0;

        // Create new sub-sequences
        for (uint_t workIdx = 0; workIdx < oldHostWorkCounter; workIdx++) {
            h_glob_seq_t hostParams = h_hostGlobalParams[workIdx];
            d_glob_seq_t devParams = h_devGlobalParams[workIdx];

            // New subsequece (lower)
            if (devParams.offsetLower > MIN_PARTITION_SIZE_GLOBAL) {
                h_hostGlobalBuffer[hostWorkCounter++].setLowerSeq(hostParams, devParams);
            } else {
                h_localParams[localWorkCounter++].setLowerSeq(hostParams, devParams);
            }

            // New subsequece (greater)
            if (devParams.offsetLower > MIN_PARTITION_SIZE_GLOBAL) {
                h_hostGlobalBuffer[hostWorkCounter++].setGreaterSeq(hostParams, devParams);
            } else {
                h_localParams[localWorkCounter++].setGreaterSeq(hostParams, devParams);
            }

            workTotal++;
        }

        h_glob_seq_t *temp = h_hostGlobalParams;
        h_hostGlobalParams = h_hostGlobalBuffer;
        h_hostGlobalBuffer = temp;
    }

    // Add sequences which were not partitioned to min size
    for (uint_t workIdx = 0; workIdx < hostWorkCounter; workIdx++) {
        h_localParams[localWorkCounter++].setFromGlobalSeq(h_hostGlobalParams[workIdx]);
    }

    runQuickSortLocalKernel(dataInput, dataBuffer, h_localParams, d_localParams, tableLen, workTotal, orderAsc);
}

void sortParallel(el_t *h_dataInput, el_t *h_dataOutput, uint_t tableLen, bool orderAsc) {
    // Data memory
    el_t *d_dataInput, *d_dataBuffer;
    // Sequences metadata for GLOBAL quicksort on HOST
    h_glob_seq_t *h_globalSeqHost, *h_globalSeqHostBuffer;
    // Sequences metadata for GLOBAL quicksort on DEVICE
    d_glob_seq_t *h_globalSeqDev, *d_globalSeqDev;
    // Array of sequence indexes for thread blocks in GLOBAL quicksort. This way thread blocks know which
    // sequence they have to partition.
    uint_t *h_globalSeqIndexes, *d_globalSeqIndexes;
    // Sequences metadata for LOCAL quicksort
    loc_seq_t *h_localSeq, *d_localSeq;

    // Maximum number of sequneces which can get generated by global quicksort. In global quicksort sequences
    // are generated untill total number of sequences is lower than tableLen / MIN_PARTITION_SIZE_GLOBAL.
    uint_t maxSequences = 2 * tableLen / MIN_PARTITION_SIZE_GLOBAL - 2;
    // Max number of all thread blocks in GLOBAL quicksort. TODO verify constant 2.
    uint_t maxNumThreadBlocks = 2 * tableLen / (THREADS_PER_SORT_GLOBAL * ELEMENTS_PER_THREAD_GLOBAL);

    LARGE_INTEGER timer;
    cudaError_t error;

    memoryInitHost(
        &h_globalSeqHost, &h_globalSeqHostBuffer, &h_globalSeqDev, &h_globalSeqIndexes, &h_localSeq,
        maxSequences, maxNumThreadBlocks
    );
    memoryInitDevice(
        h_dataInput, &d_dataInput, &d_dataBuffer, &d_globalSeqDev, &d_globalSeqIndexes, &d_localSeq,
        tableLen, maxSequences, maxNumThreadBlocks
    );

    startStopwatch(&timer);
    quickSort(
        h_dataInput, d_dataInput, d_dataBuffer, h_globalSeqHost, h_globalSeqHostBuffer, h_globalSeqDev,
        d_globalSeqDev, h_globalSeqIndexes, d_globalSeqIndexes, h_localSeq, d_localSeq, tableLen, orderAsc
    );

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    double time = endStopwatch(timer, "Executing parallel quicksort.");
    printf("Operations (pair swaps): %.2f M/s\n", tableLen / 500.0 / time);

    error = cudaMemcpy(h_dataOutput, d_dataBuffer, tableLen * sizeof(*h_dataOutput), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    cudaFree(d_dataInput);
    cudaFree(d_dataBuffer);
}
