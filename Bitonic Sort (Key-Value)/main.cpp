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


int main(int argc, char** argv) {
    data_t *h_inputKeys, *h_inputValues;
    data_t *h_outputParallelKeys, *h_outputParallelValues;
    data_t *h_outputSequentialKeys, *h_outputSequentialValues;
    data_t *h_outputCorrect;
    data_t *d_dataTableKeys, *d_dataTableValues;
    double **timers;

    uint_t tableLen = (1 << 10);
    uint_t interval = (1 << 31);
    uint_t testRepetitions = 10;    // How many times are sorts ran
    order_t sortOrder = ORDER_ASC;  // Values: ORDER_ASC, ORDER_DESC
    data_dist_t distribution = DISTRIBUTION_UNIFORM;
    bool printMeaurements = false;

    // Designates whether paralle/sequential sort has always sorted data correctly. NOT CONFIGURABLE!
    bool parallelSortsCorrectly = true, sequentialSortsCorrectly = true;
    cudaError_t error;

    // Memory alloc
    allocHostMemory(
        &h_inputKeys, &h_inputValues, &h_outputParallelKeys, &h_outputParallelValues, &h_outputSequentialKeys,
        &h_outputSequentialValues, &h_outputCorrect, &timers, tableLen, testRepetitions
    );
    allocDeviceMemory(&d_dataTableKeys, &d_dataTableValues, tableLen);

    printf(">>> BITONIC SORT <<<\n\n\n");
    printDataDistribution(distribution);
    if (printMeaurements)
    {
        printf("\n");
        printTableHeaderKeysOnly();
    }

    for (uint_t i = 0; i < testRepetitions; i++)
    {
        fillTableKeysOnly(h_inputKeys, tableLen, interval, distribution);

        // Sort parallel
        error = cudaMemcpy(
            d_dataTableKeys, h_inputKeys, tableLen * sizeof(*d_dataTableKeys), cudaMemcpyHostToDevice
        );
        checkCudaError(error);
        error = cudaMemcpy(
            d_dataTableValues, h_inputValues, tableLen * sizeof(*d_dataTableValues), cudaMemcpyHostToDevice
        );
        checkCudaError(error);
        timers[SORT_PARALLEL][i] = sortParallel(
            h_inputKeys, h_outputParallelKeys, d_dataTableKeys, tableLen, sortOrder
        );

        // Sort sequential
        std::copy(h_inputKeys, h_inputKeys + tableLen, h_outputSequentialKeys);
        timers[SORT_SEQUENTIAL][i] = sortSequential(h_outputSequentialKeys, tableLen, sortOrder);

        // Sort correct
        std::copy(h_inputKeys, h_inputKeys + tableLen, h_outputCorrect);
        timers[SORT_CORRECT][i] = sortCorrect(h_outputCorrect, tableLen, sortOrder);

        bool areEqualParallel = compareArrays(h_outputParallelKeys, h_outputCorrect, tableLen);
        bool areEqualSequential = compareArrays(h_outputSequentialKeys, h_outputCorrect, tableLen);

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
    freeHostMemory(
        h_inputKeys, h_inputValues, h_outputParallelKeys, h_outputParallelValues, h_outputSequentialKeys,
        h_outputSequentialValues, h_outputCorrect, timers
    );
    freeDeviceMemory(d_dataTableKeys, d_dataTableValues);

    getchar();
    return 0;
}
