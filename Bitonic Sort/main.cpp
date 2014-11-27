#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <array>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"
#include "../Utils/cuda.h"
#include "../Utils/statistics.h"
#include "constants.h"
#include "memory.h"
#include "sort_parallel.h"
#include "sort_sequential.h"


int main(int argc, char** argv) {
    data_t *h_input;
    data_t *h_outputParallel, *h_outputSequential, *h_outputCorrect, *d_dataTable;
    double **timers;

    uint_t tableLen = (1 << 20);
    uint_t interval = (1 << 31);
    uint_t testRepetitions = 30;    // How many times are sorts ran
    order_t sortOrder = ORDER_ASC;  // Values: ORDER_ASC, ORDER_DESC

    // Designates whether paralle/sequential sort has always sorted data correctly
    bool parallelSortsCorrectly = true, sequentialSortsCorrectly = true;
    cudaError_t error;

    cudaFree(NULL);     // Initializes CUDA, because CUDA init is lazy
    srand(time(NULL));  // TODO check if needed

    // Memory alloc
    allocHostMemory(
        &h_input, &h_outputParallel, &h_outputSequential, &h_outputCorrect, &timers, tableLen, testRepetitions
    );
    allocDeviceMemory(&d_dataTable, tableLen);
    printTableHeaderKeysOnly("BITONIC SORT");

    for (uint_t i = 0; i < testRepetitions; i++)
    {
        fillTableKeysOnly(h_input, tableLen, interval);

        // Sort parallel
        error = cudaMemcpy(d_dataTable, h_input, tableLen * sizeof(*d_dataTable), cudaMemcpyHostToDevice);
        checkCudaError(error);
        timers[SORT_PARALLEL][i] = sortParallel(h_input, h_outputParallel, d_dataTable, tableLen, sortOrder);

        // Sort sequential
        std::copy(h_input, h_input + tableLen, h_outputSequential);
        timers[SORT_SEQUENTIAL][i] = 99999;
        // TODO

        // Sort correct
        std::copy(h_input, h_input + tableLen, h_outputCorrect);
        timers[SORT_CORRECT][i] = sortCorrect(h_input, h_outputCorrect, tableLen);

        bool areEqualParallel = compareArrays(h_outputParallel, h_outputCorrect, tableLen);
        bool areEqualSequential = false;  // TODO

        parallelSortsCorrectly &= areEqualParallel;
        sequentialSortsCorrectly &= areEqualSequential;

        printTableLineKeysOnly(timers, i, tableLen, areEqualParallel, areEqualSequential);
    }

    printTableSplitterKeysOnly();

    // Print-out of statistics for PARALLEL sort
    printf("\n\n- PARALLEL SORT\n");
    printStatisticsKeysOnly(timers[SORT_PARALLEL], testRepetitions, tableLen, parallelSortsCorrectly);

    // Print-out of statistics for SEQUENTIAL sort
    printf("\n\n- SEQUENTIAL SORT\n");
    printStatisticsKeysOnly(timers[SORT_SEQUENTIAL], testRepetitions, tableLen, sequentialSortsCorrectly);

    printf(
        "\n\n- Speedup (SEQUENTIAL/PARALLEL): %.2lf\n",
        getSpeedup(timers, SORT_SEQUENTIAL, SORT_PARALLEL, testRepetitions)
    );
    printf(
        "- Speedup (CORRECT/PARALLEL):    %.2lf\n",
        getSpeedup(timers, SORT_CORRECT, SORT_PARALLEL, testRepetitions)
    );

    // Memory free
    freeHostMemory(h_input, h_outputParallel, h_outputSequential, h_outputCorrect, timers);
    freeDeviceMemory(d_dataTable);

    getchar();
    return 0;
}
