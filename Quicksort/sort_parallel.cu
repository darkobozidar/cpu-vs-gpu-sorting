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
Initializes memory needed for paralel sort implementation.
*/
void memoryInit(el_t *h_input, el_t **d_dataInput, el_t **d_dataBuffer, gparam_t **d_globalParams,
                lparam_t **d_localParams, uint_t tableLen) {
    cudaError_t error;

    error = cudaMalloc(d_dataInput, tableLen * sizeof(**d_dataInput));
    checkCudaError(error);
    error = cudaMalloc(d_dataBuffer, tableLen * sizeof(**d_dataBuffer));
    checkCudaError(error);
    error = cudaMalloc(d_globalParams, 2 * MAX_SEQUENCES * sizeof(**d_globalParams));
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
void quickSort(el_t *dataInput, el_t *dataBuffer, gparam_t *h_globalParams, gparam_t *d_globalParams,
               lparam_t *h_localParams, lparam_t *d_localParams, uint_t tableLen, bool orderAsc) {
    h_globalParams[0].start = 0;
    h_globalParams[0].length = tableLen;
    h_globalParams[0].oldStart = 0;
    h_globalParams[0].oldLength = tableLen;
    h_globalParams[0].direction = false;

    uint_t workCounter = 1;
    uint_t minLength = tableLen / MAX_SEQUENCES;

    while (workCounter < MAX_SEQUENCES) {
        uint_t threadBlockSize = 0;

        // Because pivots are not sorted, total size of sequence gets shorter and shorter
        for (uint_t i = 0; i < workCounter; i++) {
            threadBlockSize += h_globalParams[i].length;
        }
        threadBlockSize /= MAX_SEQUENCES;

        uint_t deviceWorkCounter = 0;

        for (uint_t i = 0; i < workCounter; i++) {
            gparam_t params = h_globalParams[i];
            if (params.length <= minLength) {
                // TODO add to list "done"
                continue;
            }

            uint_t threadBlockCount = max(params.length / threadBlockSize, 1);
            for (uint_t j = 0; j < threadBlockCount; j++) {
                d_globalParams[j].start = params.start + j * threadBlockSize;
                d_globalParams[j].length = threadBlockSize;
                deviceWorkCounter++;
            }
            d_globalParams[deviceWorkCounter - 1].start = params.start + (threadBlockCount - 1) * threadBlockSize;
            d_globalParams[deviceWorkCounter - 1].length = params.length - (threadBlockCount - 1) * threadBlockCount;
        }
    }

    /*cudaMemcpy(d_localParams, h_localParams, MAX_SEQUENCES * sizeof(*d_localParams), cudaMemcpyHostToDevice);
    runQuickSortLocalKernel(dataInput, dataBuffer, d_localParams, tableLen, orderAsc);*/
}

void sortParallel(el_t *h_dataInput, el_t *h_dataOutput, uint_t tableLen, bool orderAsc) {
    el_t *d_dataInput, *d_dataBuffer;
    gparam_t h_globalParams[2 * MAX_SEQUENCES], *d_globalParams;
    lparam_t h_localParams[2 * MAX_SEQUENCES], *d_localParams;

    LARGE_INTEGER timer;
    cudaError_t error;

    memoryInit(h_dataInput, &d_dataInput, &d_dataBuffer, &d_globalParams, &d_localParams, tableLen);

    startStopwatch(&timer);
    quickSort(d_dataInput, d_dataBuffer, h_globalParams, d_globalParams, h_localParams, d_localParams,
              tableLen, orderAsc);

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    double time = endStopwatch(timer, "Executing parallel quicksort.");
    printf("Operations (pair swaps): %.2f M/s\n", tableLen / 500.0 / time);

    error = cudaMemcpy(h_dataOutput, d_dataBuffer, tableLen * sizeof(*h_dataOutput), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    cudaFree(d_dataInput);
    cudaFree(d_dataBuffer);
}
