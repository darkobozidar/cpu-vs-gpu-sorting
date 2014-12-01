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
#include "../Utils/generator.h"
#include "../Utils/sort_correct.h"
#include "../Utils/statistics.h"
#include "constants.h"
#include "memory.h"
#include "sort_parallel.h"
#include "sort_sequential.h"


int main(int argc, char **argv) {
    data_t *h_input;
    data_t *h_outputParallel, *h_outputSequential, *h_outputCorrect, *d_dataTable;
    double **timers;

    uint_t tableLen = (1 << 20);
    uint_t interval = (1 << 31);
    uint_t testRepetitions = 10;    // How many times are sorts ran
    order_t sortOrder = ORDER_ASC;  // Values: ORDER_ASC, ORDER_DESC
    data_dist_t distribution = DISTRIBUTION_UNIFORM;
    bool printMeaurements = false;

    // Determines whether paralle/sequential sort has always sorted data correctly. NOT CONFIGURABLE!
    bool parallelSortsCorrectly = true, sequentialSortsCorrectly = true;
    cudaError_t error;

    // Memory alloc
    allocHostMemory(
        &h_input, &h_outputParallel, &h_outputSequential, &h_outputCorrect, &timers, tableLen, testRepetitions
    );
    allocDeviceMemory(&d_dataTable, tableLen);

    printf(">>> BITONIC SORT <<<\n\n\n");
    printDataDistribution(distribution);
    printf("> Array length: %d\n", tableLen);
    if (printMeaurements)
    {
        printf("\n");
        printTableHeaderKeysOnly();
    }

    for (uint_t i = 0; i < testRepetitions; i++)
    {
        fillTableKeysOnly(h_input, tableLen, interval, distribution);

        // Sort parallel
        error = cudaMemcpy(d_dataTable, h_input, tableLen * sizeof(*d_dataTable), cudaMemcpyHostToDevice);
        checkCudaError(error);
        timers[SORT_PARALLEL][i] = 9999;  // sortParallel(h_outputParallel, d_dataTable, tableLen, sortOrder);

        // Sort sequential
        std::copy(h_input, h_input + tableLen, h_outputSequential);
        timers[SORT_SEQUENTIAL][i] = 9999;  // sortSequential(h_outputSequential, tableLen, sortOrder);

        // Sort correct
        std::copy(h_input, h_input + tableLen, h_outputCorrect);
        timers[SORT_CORRECT][i] = 9999;  // sortCorrect(h_outputCorrect, tableLen, sortOrder);

        bool areEqualParallel = compareArrays(h_outputParallel, h_outputCorrect, tableLen);
        bool areEqualSequential = compareArrays(h_outputSequential, h_outputCorrect, tableLen);

        parallelSortsCorrectly &= areEqualParallel;
        sequentialSortsCorrectly &= areEqualSequential;

        if (printMeaurements)
        {
            printTableLineKeysOnly(timers, i, tableLen, areEqualParallel, areEqualSequential);
        }
    }

    if (printMeaurements)
    {
        printTableSplitterKeysOnly();
    }

    // Print-out of statistics for PARALLEL sort
    printf("\n- PARALLEL SORT\n");
    printStatisticsKeysOnly(timers[SORT_PARALLEL], testRepetitions, tableLen, parallelSortsCorrectly);

    // Print-out of statistics for SEQUENTIAL sort
    printf("\n- SEQUENTIAL SORT\n");
    printStatisticsKeysOnly(timers[SORT_SEQUENTIAL], testRepetitions, tableLen, sequentialSortsCorrectly);

    printf(
        "\n- Speedup (SEQUENTIAL/PARALLEL): %.2lf\n",
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
