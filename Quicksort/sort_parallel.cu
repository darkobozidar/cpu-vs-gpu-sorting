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
void memoryInit() {
    cudaError_t error;
    // TODO
}

void sortParallel(el_t *h_input, el_t *h_output, uint_t tableLen, bool orderAsc) {
    LARGE_INTEGER timer;
    cudaError_t error;

    startStopwatch(&timer);

    // TODO

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    double time = endStopwatch(timer, "Executing parallel radix sort.");
    printf("Operations (pair swaps): %.2f M/s\n", tableLen / 500.0 / time);

    /*error = cudaMemcpy(h_output, d_table, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);*/

    /*cudaFree(d_table);
    cudaFree(d_bufffer);*/
}
