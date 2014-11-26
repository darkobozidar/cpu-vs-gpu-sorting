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
    data_t *h_outputParallel, *h_outputSequential, *h_outputCorrect;
    data_t *d_dataTable;

    uint_t tableLen = (1 << 10);
    uint_t interval = 1 << 31;
    uint_t testRepetitions = 1;     // How many times are sorts ran
    order_t sortOrder = ORDER_ASC;  // Values: ORDER_ASC, ORDER_DESC
    cudaError_t error;

    cudaFree(NULL);     // Initializes CUDA, because CUDA init is lazy
    srand(time(NULL));  // TODO check if needed

    // Memory alloc
    allocHostMemory(&h_input, &h_outputParallel, &h_outputSequential, &h_outputCorrect, tableLen);
    allocDeviceMemory(&d_dataTable, tableLen);

    for (uint_t i = 0; i < testRepetitions; i++)
    {
        fillTableKey(h_input, tableLen, interval);

        // Sort parallel
        error = cudaMemcpy(d_dataTable, h_input, tableLen * sizeof(*d_dataTable), cudaMemcpyHostToDevice);
        checkCudaError(error);
        sortParallel(h_input, h_outputParallel, d_dataTable, tableLen, sortOrder);

        // Sort sequential
        std::copy(h_input, h_input + tableLen, h_outputSequential);
        // TODO

        // Sort correct
        std::copy(h_input, h_input + tableLen, h_outputCorrect);
        sortCorrect(h_input, h_outputCorrect, tableLen);

        compareArrays(h_outputParallel, h_outputCorrect, tableLen);
    }

    // Memory free
    freeHostMemory(h_input, h_outputParallel, h_outputSequential, h_outputCorrect);
    freeDeviceMemory(d_dataTable);

    getchar();
    return 0;
}
