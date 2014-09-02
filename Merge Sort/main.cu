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


int main(int argc, char** argv) {
    // Rename array to table everywhere in code
    /*data_t inputData[32] = {
        6, 23, 29, 35, 45, 63, 64, 97, 1, 4, 25, 34, 45, 67, 98, 99, 4, 19, 41, 58,
        68, 80, 81, 96, 4, 13, 18, 33, 55, 66, 88, 90
    };*/
    /*data_t inputData[64] = {
        60, 39, 36, 61, 40, 41, 62, 54, 42, 64, 81, 70, 55, 5, 99, 22, 49, 95, 18, 19,
        73, 84, 90, 16, 50, 22, 1, 60, 6, 74, 58, 18, 43, 64, 18, 86, 33, 81, 92, 42,
        14, 81, 34, 37, 43, 29, 12, 30, 81, 41, 21, 8, 82, 45, 40, 25, 96, 85, 25, 32,
        90, 88, 20, 28
    };*/
    data_t* inputData;
    data_t* outputDataParallel;
    data_t* outputDataSequential;
    data_t* outputDataCorrect;

    uint_t dataLen = 1 << 20;
    bool orderAsc = true;  // TODO use this
    cudaError_t error;

    LARGE_INTEGER timerStart;

    // TODO remove bottom comment when tested
    //cudaFuncCachePreferNone, cudaFuncCachePreferShared, cudaFuncCachePreferL1, cudaFuncCachePreferEqual
    error = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    checkCudaError(error);
    cudaFree(NULL);  // Initializes CUDA, because CUDA init is lazy
    srand(time(NULL));

    error = cudaHostAlloc(&inputData, dataLen * sizeof(*inputData), cudaHostAllocDefault);
    checkCudaError(error);

    for (int i = 0; i < 3; i++) {
        fillArrayRand(inputData, dataLen);
        //fillArrayValue(inputData, dataLen, 5);
        //printArray(inputData, dataLen);

        outputDataParallel = sortParallel(inputData, dataLen, orderAsc);
        //printArray(outputDataParallel, dataLen);

        outputDataCorrect = sortCorrect(inputData, dataLen);
        compareArrays(outputDataParallel, outputDataCorrect, dataLen);
    }

    //cudaFreeHost(inputData);
    cudaFreeHost(outputDataParallel);
    //free(outputDataSequential);
    free(outputDataCorrect);

    getchar();
    return 0;
}
