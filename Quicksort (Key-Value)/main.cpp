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


int main(int argc, char **argv)
{
    data_t *h_inputKeys, *h_inputValues;
    data_t *h_outputParallelKeys, *h_outputParallelValues;
    data_t *h_outputSequentialKeys, *h_outputSequentialValues;
    data_t *h_outputCorrect;
    data_t *d_dataKeys, *d_dataValues, *d_bufferKeys, *d_bufferValues;
    // When initial min/max parallel reduction reduces data to threashold, min/max values are coppied to host
    // and reduction is finnished on host. Multiplier "2" is used because of min and max values.
    data_t *h_minMaxValues;
    // Sequences metadata for GLOBAL quicksort on HOST
    h_glob_seq_t *h_globalSeqHost, *h_globalSeqHostBuffer;
    // Sequences metadata for GLOBAL quicksort on DEVICE
    d_glob_seq_t *h_globalSeqDev, *d_globalSeqDev;
    // Array of sequence indexes for thread blocks in GLOBAL quicksort. This way thread blocks know which
    // sequence they have to partition.
    uint_t *h_globalSeqIndexes, *d_globalSeqIndexes;
    // Sequences metadata for LOCAL quicksort
    loc_seq_t *h_localSeq, *d_localSeq;
    double **timers;

    uint_t tableLen = (1 << 20);
    uint_t interval = (1 << 31);
    uint_t testRepetitions = 10;    // How many times are sorts ran
    order_t sortOrder = ORDER_ASC;  // Values: ORDER_ASC, ORDER_DESC
    data_dist_t distribution = DISTRIBUTION_UNIFORM;
    bool printMeaurements = true;

    // Determines whether paralle/sequential sort has always sorted data correctly. NOT CONFIGURABLE!
    bool sortsCorrectlyParallel = true, sortsCorrectlySequential = true;
    // Determines whether paralle/sequential sort has always been stable. NOT CONFIGURABLE!
    bool isStableParallel = true, isStableSequential = true;
    cudaError_t error;

    // Memory alloc
    allocHostMemory(
        &h_inputKeys, &h_inputValues, &h_outputParallelKeys, &h_outputParallelValues, &h_outputSequentialKeys,
        &h_outputSequentialValues, &h_outputCorrect, &h_minMaxValues, &h_globalSeqHost, &h_globalSeqHostBuffer,
        &h_globalSeqDev, &h_globalSeqIndexes, &h_localSeq, &timers, tableLen, testRepetitions
    );
    allocDeviceMemory(
        &d_dataKeys, &d_dataValues, &d_bufferKeys, &d_bufferValues, &d_globalSeqDev, &d_globalSeqIndexes,
        &d_localSeq, tableLen
    );

    printf(">>> QUICKSORT (Key-Value) <<<\n\n\n");
    printDataDistribution(distribution);
    printf("> Array length: %d\n", tableLen);
    if (printMeaurements)
    {
        printf("\n");
        printTableHeaderKeyValue();
    }

    for (uint_t i = 0; i < testRepetitions; i++)
    {
        fillTableKeyValue(h_inputKeys, h_inputValues, tableLen, interval, distribution);

        // Sort parallel
        error = cudaMemcpy(
            d_dataKeys, h_inputKeys, tableLen * sizeof(*d_dataKeys), cudaMemcpyHostToDevice
        );
        checkCudaError(error);
        error = cudaMemcpy(
            d_dataValues, h_inputValues, tableLen * sizeof(*d_dataValues), cudaMemcpyHostToDevice
        );
        checkCudaError(error);
        error = cudaDeviceSynchronize();
        checkCudaError(error);
        timers[SORT_PARALLEL][i] = 999;  /*sortParallel(
            h_outputParallelKeys, h_outputParallelValues, d_dataKeys, d_dataValues, tableLen, sortOrder
        );*/

        // Sort sequential
        std::copy(h_inputKeys, h_inputKeys + tableLen, h_outputSequentialKeys);
        std::copy(h_inputValues, h_inputValues + tableLen, h_outputSequentialValues);
        timers[SORT_SEQUENTIAL][i] = 999;  /*sortSequential(
            h_outputSequentialKeys, h_outputSequentialValues, tableLen, sortOrder
        );*/

        // Sort correct
        std::copy(h_inputKeys, h_inputKeys + tableLen, h_outputCorrect);
        timers[SORT_CORRECT][i] = sortCorrect(h_outputCorrect, tableLen, sortOrder);

        bool sortsCorrectlyParallelLocal = compareArrays(h_outputParallelKeys, h_outputCorrect, tableLen);
        bool sortsCorrectlySequentialLocal = compareArrays(h_outputSequentialKeys, h_outputCorrect, tableLen);
        bool isStableParallelLocal = isSortStable(h_outputParallelKeys, h_outputParallelValues, tableLen);
        bool isStableSequentialLocal = isSortStable(h_outputSequentialKeys, h_outputSequentialValues, tableLen);

        sortsCorrectlyParallel &= sortsCorrectlyParallelLocal;
        sortsCorrectlySequential &= sortsCorrectlySequentialLocal;
        isStableParallel &= isStableParallelLocal;
        isStableSequential &= isStableSequentialLocal;

        if (printMeaurements)
        {
            printTableLineKeyValue(
                timers, i, tableLen, sortsCorrectlyParallelLocal, sortsCorrectlySequentialLocal,
                isStableParallelLocal, isStableSequentialLocal
            );
        }
    }

    if (printMeaurements)
    {
        printTableSplitterKeyValue();
    }

    // Print-out of statistics for PARALLEL sort
    printf("\n- PARALLEL SORT\n");
    printStatisticsKeyValue(
        timers[SORT_PARALLEL], testRepetitions, tableLen, sortsCorrectlyParallel, isStableParallel
    );

    // Print-out of statistics for SEQUENTIAL sort
    printf("\n- SEQUENTIAL SORT\n");
    printStatisticsKeyValue(
        timers[SORT_SEQUENTIAL], testRepetitions, tableLen, sortsCorrectlySequential, isStableSequential
    );

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
        h_outputSequentialValues, h_outputCorrect, h_minMaxValues, h_globalSeqHost, h_globalSeqHostBuffer,
        h_globalSeqDev, h_globalSeqIndexes, h_localSeq, timers
    );
    freeDeviceMemory(
        d_dataKeys, d_dataValues, d_bufferKeys, d_bufferValues, d_globalSeqDev, d_globalSeqIndexes, d_localSeq
    );

    getchar();
    return 0;
}
