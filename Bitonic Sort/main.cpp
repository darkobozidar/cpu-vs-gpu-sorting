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
#include "constants.h"
#include "memory.h"
#include "sort_parallel.h"
#include "sort_sequential.h"


int main(int argc, char** argv) {
    data_t *h_input;
    data_t *h_outputParallel, *h_outputSequential, *h_outputCorrect, *d_dataTable;
    double **timers;

    uint_t tableLen = (1 << 16);
    uint_t interval = (1 << 31);
    uint_t testRepetitions = 10;     // How many times are sorts ran
    order_t sortOrder = ORDER_ASC;  // Values: ORDER_ASC, ORDER_DESC
    cudaError_t error;

    cudaFree(NULL);     // Initializes CUDA, because CUDA init is lazy
    srand(time(NULL));  // TODO check if needed

    // Memory alloc
    allocHostMemory(
        &h_input, &h_outputParallel, &h_outputSequential, &h_outputCorrect, &timers, tableLen, testRepetitions
    );
    allocDeviceMemory(&d_dataTable, tableLen);

    printf("======================================================================================================\n");
    printf("||                                           BITONIC SORT                                           ||\n");
    printf("======================================================================================================\n");
    printf("||     # ||              PARALLEL              ||              SEQUENTIAL            ||   CORRECT   ||\n");
    printf("======================================================================================================\n");
    printf("||     # ||     time    |      rate     |  OK  ||     time    |      rate     |  OK  ||     time    ||\n");
    printf("======================================================================================================\n");

    for (uint_t i = 0; i < testRepetitions; i++)
    {
        fillTableKey(h_input, tableLen, interval);

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

        printf(
            "|| %5d || %8.2lf ms | %8.2lf el/s | %s  || %8.2lf ms | %8.2lf el/s | %s  || %8.2lf ms ||\n", i,
            timers[SORT_PARALLEL][i], tableLen / 1000.0 / timers[SORT_PARALLEL][i], areEqualParallel ? "YES" : "NO ",
            timers[SORT_SEQUENTIAL][i], tableLen / 1000.0 / timers[SORT_SEQUENTIAL][i], areEqualSequential ? "YES" : "NO ",
            timers[SORT_CORRECT][i]
        );
    }

    printf("======================================================================================================\n");

    // Memory free
    freeHostMemory(h_input, h_outputParallel, h_outputSequential, h_outputCorrect, timers);
    freeDeviceMemory(d_dataTable);

    getchar();
    return 0;
}
