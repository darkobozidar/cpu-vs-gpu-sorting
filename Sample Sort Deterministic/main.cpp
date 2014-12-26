#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_types.h"
#include "constants.h"
#include "utils_cuda.h"
#include "utils_host.h"
#include "sort_parallel.h"
#include "sort_sequential.h"


int main(int argc, char** argv)
{
    el_t *input;
    /*el_t input[16] = {
        2, 0, 3, 1, 27, 2, 12, 3, 58, 4, 45, 5, 95, 6, 25, 7, 67,
        8, 31, 9, 46, 10, 76, 11, 24, 12, 74, 13, 86, 14, 19, 15
    };*/
    /*el_t input[32] = {
        61, 0, 24, 1, 14, 2, 13, 3, 85, 4, 67, 5, 9, 6, 13, 7, 16, 8, 28, 9, 81, 10,
        25, 11, 57, 12, 17, 13, 32, 14, 43, 15, 48, 16, 21, 17, 31, 18, 22, 19, 83, 20,
        78, 21, 73, 22, 29, 23, 42, 24, 62, 25, 4, 26, 43, 27, 60, 28, 48, 29, 68, 30, 52, 31
    };*/
    el_t *outputParallel;
    el_t *outputCorrect;

    uint_t tableLen = (1 << 20);
    uint_t interval = 1 << 31;
    order_t sortOrder = ORDER_ASC;  // Values: ORDER_ASC, ORDER_DESC
    cudaError_t error;

    cudaFree(NULL);  // Initializes CUDA, because CUDA init is lazy

    input = (el_t*)malloc(tableLen * sizeof(*input));
    checkMallocError(input);
    outputParallel = (el_t*)malloc(tableLen * sizeof(*outputParallel));
    checkMallocError(outputParallel);

    for (uint_t i = 0; i < 1; i++) {
        fillTable(input, tableLen, interval);
        /*printTable(input, tableLen);*/
        sortParallel(input, outputParallel, tableLen, sortOrder);
    }

    /*printTable(outputParallel, tableLen);*/

    printf("\n");
    outputCorrect = sortCorrect(input, tableLen);
    compareArrays(outputParallel, outputCorrect, tableLen);

    ////cudaFreeHost(inputData);
    cudaFreeHost(outputParallel);
    ////free(outputDataSequential);
    //free(outputDataCorrect);

    getchar();
    return 0;
}
