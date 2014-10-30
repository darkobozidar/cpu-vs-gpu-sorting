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
                    uint_t maxNumSequences, uint_t maxNumThreadBlocks) {
    // TODO malloc in pinned memory
    *h_globalSeqHost = new h_glob_seq_t[maxNumSequences];
    *h_globalSeqHostBuffer = new h_glob_seq_t[maxNumSequences];
    *h_globalSeqDev = new d_glob_seq_t[maxNumSequences];
    *h_globalSeqIndexes = new uint_t[maxNumThreadBlocks];
    *h_localSeq = new loc_seq_t[maxNumSequences];
}

/*
Initializes DEVICE memory needed for paralel sort implementation.
*/
void memoryInitDevice(el_t *h_input, el_t **d_dataInput, el_t **d_dataBuffer, d_glob_seq_t **d_globalSeqDev,
                      uint_t **d_globalSeqIndexes, loc_seq_t **d_localSeq, uint_t tableLen,
                      uint_t maxNumSequences, uint_t maxNumThreadBlocks) {
    cudaError_t error;

    // Data memory allocation
    error = cudaMalloc(d_dataInput, tableLen * sizeof(**d_dataInput));
    checkCudaError(error);
    error = cudaMalloc(d_dataBuffer, tableLen * sizeof(**d_dataBuffer));
    checkCudaError(error);
    // Sequence metadata memory allocation
    error = cudaMalloc(d_globalSeqDev, maxNumSequences * sizeof(**d_globalSeqDev));
    checkCudaError(error);
    error = cudaMalloc(d_globalSeqIndexes, maxNumThreadBlocks * sizeof(**d_globalSeqIndexes));
    checkCudaError(error);
    error = cudaMalloc(d_localSeq, maxNumSequences * sizeof(**d_localSeq));
    checkCudaError(error);

    error = cudaMemcpy(*d_dataInput, h_input, tableLen * sizeof(**d_dataInput), cudaMemcpyHostToDevice);
    checkCudaError(error);
}

void runQuickSortGlobalKernel(el_t *dataInput, el_t* dataBuffer, d_glob_seq_t *h_globalSeqHost,
                              d_glob_seq_t *d_globalSeqHost, uint_t *h_globalSeqIndexes, uint_t *d_globalSeqIndexes,
                              uint_t numSeqGlobal, uint_t threadBlockCounter, uint_t tableLen) {
    cudaError_t error;
    LARGE_INTEGER timer;

    // TODO comment shared memory size, 2 * size should be enough, because scan and min/max can be
    // performed in the same array
    // TODO comment: max(min/max, lower/greater)
    uint_t sharedMemSize = max(
        2 * THREADS_PER_SORT_GLOBAL * sizeof(data_t), 2 * THREADS_PER_SORT_GLOBAL * sizeof(uint_t)
    );
    dim3 dimGrid(threadBlockCounter, 1, 1);
    dim3 dimBlock(THREADS_PER_SORT_GLOBAL, 1, 1);

    startStopwatch(&timer);

    error = cudaMemcpy(d_globalSeqHost, h_globalSeqHost, numSeqGlobal * sizeof(*d_globalSeqHost),
                       cudaMemcpyHostToDevice);
    checkCudaError(error);
    error = cudaMemcpy(d_globalSeqIndexes, h_globalSeqIndexes, threadBlockCounter * sizeof(*d_globalSeqIndexes),
                       cudaMemcpyHostToDevice);
    checkCudaError(error);

    quickSortGlobalKernel<<<dimGrid, dimBlock, sharedMemSize>>>(
        dataInput, dataBuffer, d_globalSeqHost, d_globalSeqIndexes, tableLen
    );

    error = cudaMemcpy(h_globalSeqHost, d_globalSeqHost, numSeqGlobal * sizeof(*h_globalSeqHost),
                       cudaMemcpyDeviceToHost);
    checkCudaError(error);

    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing global parallel quicksort.");*/
}

void runQuickSortLocalKernel(el_t *dataInput, el_t *dataBuffer, loc_seq_t *h_localSeq, loc_seq_t *d_localSeq,
                             uint_t tableLen, uint_t numThreadBlocks, bool orderAsc) {
    cudaError_t error;
    LARGE_INTEGER timer;

    // The same shared memory array is used for counting elements greater/lower than pivot and for bitonic sort.
    // max(intra-block scan array size, array size for bitonic sort)
    uint_t sharedMemSize = max(
        2 * THREADS_PER_SORT_LOCAL * sizeof(uint_t), BITONIC_SORT_SIZE_LOCAL * sizeof(*dataInput)
    );
    dim3 dimGrid(numThreadBlocks, 1, 1);
    dim3 dimBlock(THREADS_PER_SORT_LOCAL, 1, 1);

    startStopwatch(&timer);
    error = cudaMemcpy(d_localSeq, h_localSeq, numThreadBlocks * sizeof(*d_localSeq), cudaMemcpyHostToDevice);
    checkCudaError(error);

    quickSortLocalKernel<<<dimGrid, dimBlock, sharedMemSize>>>(
        dataInput, dataBuffer, d_localSeq, tableLen, orderAsc
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

void runMinMaxReductionKernel() {

}

void minMaxReduciton(el_t *d_dataInput, el_t *d_dataBuffer, data_t &minVal, data_t &maxVal, uint_t tableLen) {
    // Number of min/max values
    uint_t numValues = tableLen;

    while (numValues < THRESHOLD_REDUCTION) {
        // Half of the array for min values and the other half for max values
        uint_t sharedMemSize = 2 * THREADS_PER_REDUCTION * sizeof(data_t);

        dim3 dimGrid((numValues - 1) / (THREADS_PER_REDUCTION * ELEMENTS_PER_THREAD_REDUCTION) + 1, 1, 1);
        dim3 dimBlock(THREADS_PER_REDUCTION, 1, 1);


        numValues = dimGrid.x;
    }
}

// TODO handle empty sub-blocks
void quickSort(el_t *h_dataInput, el_t *d_dataInput, el_t *d_dataBuffer, h_glob_seq_t *h_globalSeqHost,
               h_glob_seq_t *h_globalSeqHostBuffer, d_glob_seq_t *h_globalSeqDev, d_glob_seq_t *d_globalSeqDev,
               uint_t *h_globalSeqIndexes, uint_t *d_globalSeqIndexes, loc_seq_t *h_localSeq,
               loc_seq_t *d_localSeq, uint_t tableLen, bool orderAsc) {
    // TODO parallel reduction for initial pivot
    // TODO in global quicksort there is no need to calculate min and max after it is calculated first time
    uint_t minVal = min(min(h_dataInput[0].key, h_dataInput[tableLen / 2].key), h_dataInput[tableLen - 1].key);
    uint_t maxVal = max(max(h_dataInput[0].key, h_dataInput[tableLen / 2].key), h_dataInput[tableLen - 1].key);
    h_globalSeqHost[0].setInitSeq(tableLen, (minVal + maxVal) / 2);

    uint_t numSeqGlobal = 1; // Number of sequences for GLOBAL quicksort
    uint_t numSeqLocal = 0;  // Number of sequences for LOCAL quicksort
    uint_t numSeqLimit = (tableLen - 1) / MIN_PARTITION_SIZE_GLOBAL + 1;
    uint_t elemsPerThreadBlock = THREADS_PER_SORT_GLOBAL * ELEMENTS_PER_THREAD_GLOBAL;
    cudaError_t error;

    // TODO if statement for initial sequence length
    while (numSeqGlobal + numSeqLocal < numSeqLimit) {
        uint_t threadBlockCounter = 0;

        // Transfers host sequences to device sequences (device needs different data about sequence than host)
        for (uint_t seqIdx = 0; seqIdx < numSeqGlobal; seqIdx++) {
            uint_t threadBlocksPerSeq = (h_globalSeqHost[seqIdx].length - 1) / elemsPerThreadBlock + 1;
            h_globalSeqDev[seqIdx].setFromHostSeq(h_globalSeqHost[seqIdx], threadBlockCounter, threadBlocksPerSeq);

            // For all thread blocks in current iteration marks, they are assigned to current sequence.
            for (uint_t blockIdx = 0; blockIdx < threadBlocksPerSeq; blockIdx++) {
                h_globalSeqIndexes[threadBlockCounter++] = seqIdx;
            }
        }

        runQuickSortGlobalKernel(
            d_dataInput, d_dataBuffer, h_globalSeqDev, d_globalSeqDev, h_globalSeqIndexes,
            d_globalSeqIndexes, numSeqGlobal, threadBlockCounter, tableLen
        );

        /*printf("-----------------------------------\n");
        for (uint_t seqIdx = 0; seqIdx < numSeqGlobal; seqIdx++) {
            printf("%2d ", h_globalSeqHost[seqIdx].pivot);
        }
        printf("\n");
        runPrintTableKernel(d_dataInput, tableLen);
        runPrintTableKernel(d_dataBuffer, tableLen);
        printf("\n-----------------------------------\n");*/

        uint_t numSeqGlobalOld = numSeqGlobal;
        numSeqGlobal = 0;

        // Creates new sub-sequences
        // TODO if sequence length is > 0
        for (uint_t seqIdx = 0; seqIdx < numSeqGlobalOld; seqIdx++) {
            h_glob_seq_t seqHost = h_globalSeqHost[seqIdx];
            d_glob_seq_t seqDev = h_globalSeqDev[seqIdx];

            // New subsequece (lower)
            if (seqDev.offsetLower > MIN_PARTITION_SIZE_GLOBAL) {
                h_globalSeqHostBuffer[numSeqGlobal++].setLowerSeq(seqHost, seqDev);
            } else {
                h_localSeq[numSeqLocal++].setLowerSeq(seqHost, seqDev);
            }

            // New subsequece (greater)
            if (seqDev.offsetGreater > MIN_PARTITION_SIZE_GLOBAL) {
                h_globalSeqHostBuffer[numSeqGlobal++].setGreaterSeq(seqHost, seqDev);
            } else {
                h_localSeq[numSeqLocal++].setGreaterSeq(seqHost, seqDev);
            }
        }

        h_glob_seq_t *temp = h_globalSeqHost;
        h_globalSeqHost = h_globalSeqHostBuffer;
        h_globalSeqHostBuffer = temp;
    }

    // Adds sequences which were not partitioned by global quicksort to sequences for local quicksort
    for (uint_t seqIdx = 0; seqIdx < numSeqGlobal; seqIdx++) {
        h_localSeq[numSeqLocal++].setFromGlobalSeq(h_globalSeqHost[seqIdx]);
    }

    runQuickSortLocalKernel(
        d_dataInput, d_dataBuffer, h_localSeq, d_localSeq, tableLen, numSeqLocal, orderAsc
    );
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
    // are generated untill total number of sequences is lower than: tableLen / MIN_PARTITION_SIZE_GLOBAL.
    uint_t maxNumSequences = 2 * ((tableLen - 1) / MIN_PARTITION_SIZE_GLOBAL + 1);
    // Max number of all thread blocks in GLOBAL quicksort.
    uint_t elemsPerThreadBlock = (THREADS_PER_SORT_GLOBAL * ELEMENTS_PER_THREAD_GLOBAL);
    uint_t maxNumThreadBlocks = maxNumSequences * ((MIN_PARTITION_SIZE_GLOBAL - 1) / elemsPerThreadBlock + 1);

    LARGE_INTEGER timer;
    cudaError_t error;

    memoryInitHost(
        &h_globalSeqHost, &h_globalSeqHostBuffer, &h_globalSeqDev, &h_globalSeqIndexes, &h_localSeq,
        maxNumSequences, maxNumThreadBlocks
    );
    memoryInitDevice(
        h_dataInput, &d_dataInput, &d_dataBuffer, &d_globalSeqDev, &d_globalSeqIndexes, &d_localSeq,
        tableLen, maxNumSequences, maxNumThreadBlocks
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
