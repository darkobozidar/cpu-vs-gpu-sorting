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
Initializes memory needed for paralel sort implementation.
*/
void memoryInit(el_t *h_input, el_t **d_dataInput, el_t **d_dataBuffer, d_gparam_t **d_globalParams,
                uint_t **d_globalSeqIndexes, lparam_t **d_localParams, uint_t tableLen) {
    cudaError_t error;

    error = cudaMalloc(d_dataInput, tableLen * sizeof(**d_dataInput));
    checkCudaError(error);
    error = cudaMalloc(d_dataBuffer, tableLen * sizeof(**d_dataBuffer));
    checkCudaError(error);
    error = cudaMalloc(d_globalParams, 2 * MAX_SEQUENCES * sizeof(**d_globalParams));
    checkCudaError(error);
    error = cudaMalloc(d_globalSeqIndexes, 2 * MAX_SEQUENCES * sizeof(**d_globalSeqIndexes));
    checkCudaError(error);
    error = cudaMalloc(d_localParams, 2 * MAX_SEQUENCES * sizeof(**d_localParams));
    checkCudaError(error);

    error = cudaMemcpy(*d_dataInput, h_input, tableLen * sizeof(**d_dataInput), cudaMemcpyHostToDevice);
    checkCudaError(error);
}

void runQuickSortLocalKernel(el_t *input, el_t *output, lparam_t *localParams, uint_t tableLen, bool orderAsc) {
    cudaError_t error;
    LARGE_INTEGER timer;

    // The same shared memory array is used for counting elements greater/lower than pivot and for bitonic sort.
    // max(intra-block-scan array size, array size for bitonic sort)
    uint_t sharedMemSize = max(
        2 * THREADS_PER_SORT_LOCAL * sizeof(uint_t), BITONIC_SORT_SIZE_LOCAL * sizeof(*input)
    );
    uint_t elementsPerBlock = tableLen / MAX_SEQUENCES;
    dim3 dimGrid(MAX_SEQUENCES, 1, 1);
    dim3 dimBlock(THREADS_PER_SORT_LOCAL, 1, 1);

    startStopwatch(&timer);
    quickSortLocalKernel<<<dimGrid, dimBlock, sharedMemSize>>>(
        input, output, localParams, tableLen, orderAsc
    );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing local parallel quicksort.");*/
}

// TODO handle empty sub-blocks
void quickSort(el_t *hostData, el_t *dataInput, el_t *dataBuffer, h_gparam_t *h_hostGlobalParams,
               h_gparam_t *h_hostGlobalBuffer, d_gparam_t *h_devGlobalParams, d_gparam_t *d_devGlobalParams,
               uint_t *h_globalSeqIndexes, uint_t *d_globalSeqIndexes, lparam_t *h_localParams,
               lparam_t *d_localParams, uint_t tableLen, bool orderAsc) {
    // Set starting work
    uint_t minVal = min(min(hostData[0].key, hostData[tableLen / 2].key), hostData[tableLen - 1].key);
    uint_t maxVal = max(max(hostData[0].key, hostData[tableLen / 2].key), hostData[tableLen - 1].key);
    h_hostGlobalParams[0].setDefaultParams(tableLen);
    h_hostGlobalParams[0].pivot = (minVal + maxVal) / 2;

    // Size of workstack
    uint_t hostWorkCounter = 1;
    uint_t elemsPerThreadBlock = THREADS_PER_SORT_GLOBAL * ELEMENTS_PER_THREAD_GLOBAL;
    // Maximum number of sequences which can be generated with global quicksort
    uint_t maxSequences = tableLen / (elemsPerThreadBlock * 1);  // TODO replace 1 with constant
    cudaError_t error;

    // TODO if statement for initial sequence length
    while (hostWorkCounter < maxSequences) {
        uint_t threadBlockCounter = 0;

        // Store work to device
        for (uint_t workIdx = 0; workIdx < hostWorkCounter; workIdx++) {
            uint_t threadBlocksPerSequence = (h_hostGlobalParams[workIdx].length - 1) / elemsPerThreadBlock + 1;

            // For every thread block marks, which sequence they have to partiton (which work they have to perform)
            for (uint_t blockIdx = 0; blockIdx < threadBlocksPerSequence; blockIdx++) {
                h_globalSeqIndexes[threadBlockCounter++] = workIdx;
            }

            // Store work, that thread blocks assigned to current sequence have to perform
            h_devGlobalParams[workIdx].fromHostGlobalParams(h_hostGlobalParams[workIdx]);
        }

        cudaMemcpy(d_devGlobalParams, h_devGlobalParams, hostWorkCounter * sizeof(*d_devGlobalParams),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_globalSeqIndexes, h_globalSeqIndexes, threadBlockCounter * sizeof(*d_globalSeqIndexes),
                   cudaMemcpyHostToDevice);

        // TODO run global quicksort

        cudaMemcpy(h_devGlobalParams, d_devGlobalParams, hostWorkCounter * sizeof(*h_devGlobalParams),
                   cudaMemcpyDeviceToHost);

        // TODO generate new sequences

        break;
    }

    /*cudaMemcpy(d_localParams, h_localParams, MAX_SEQUENCES * sizeof(*d_localParams), cudaMemcpyHostToDevice);
    runQuickSortLocalKernel(dataInput, dataBuffer, d_localParams, tableLen, orderAsc);*/
}

void sortParallel(el_t *h_dataInput, el_t *h_dataOutput, uint_t tableLen, bool orderAsc) {
    el_t *d_dataInput, *d_dataBuffer;
    // Arrays needed for global quicksort
    h_gparam_t h_hostGlobalParams[2 * MAX_SEQUENCES], h_hostGlobalBuffer[2 * MAX_SEQUENCES];
    d_gparam_t h_devGlobalParams[2 * MAX_SEQUENCES], *d_devGlobalParams;
    uint_t h_globalSeqIndexes[2 * MAX_SEQUENCES], *d_globalSeqIndexes;
    // Arrays needed for local quicksort
    lparam_t h_localParams[2 * MAX_SEQUENCES], *d_localParams;

    LARGE_INTEGER timer;
    cudaError_t error;

    memoryInit(
        h_dataInput, &d_dataInput, &d_dataBuffer, &d_devGlobalParams, &d_globalSeqIndexes, &d_localParams, tableLen
    );

    startStopwatch(&timer);
    quickSort(
        h_dataInput, d_dataInput, d_dataBuffer, h_hostGlobalParams, h_hostGlobalBuffer, h_devGlobalParams,
        d_devGlobalParams, h_globalSeqIndexes, d_globalSeqIndexes, h_localParams, d_localParams, tableLen, orderAsc
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
