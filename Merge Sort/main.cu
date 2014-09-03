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
    data_t *h_inputKeys, *h_inputVals, *h_outputKeys, *h_outputVals;

    uint_t arrayLen = 1 << 5;
    uint_t interval = 65536;
    bool orderAsc = true;  // TODO use this
    cudaError_t error;

    cudaFree(NULL);  // Initializes CUDA, because CUDA init is lazy
    srand(time(NULL));

    error = cudaHostAlloc(&h_inputKeys, arrayLen * sizeof(*h_inputKeys), cudaHostAllocDefault);
    checkCudaError(error);
    error = cudaHostAlloc(&h_inputVals, arrayLen * sizeof(*h_inputVals), cudaHostAllocDefault);
    checkCudaError(error);
    error = cudaHostAlloc(&h_outputKeys, arrayLen * sizeof(*h_outputKeys), cudaHostAllocDefault);
    checkCudaError(error);
    error = cudaHostAlloc(&h_outputVals, arrayLen * sizeof(*h_outputVals), cudaHostAllocDefault);
    checkCudaError(error);

    fillArrayRand(h_inputKeys, arrayLen, interval);
    fillArrayConsecutive(h_inputVals, arrayLen);

    sortParallel(h_inputKeys, h_inputVals, h_outputKeys, h_outputVals, arrayLen, orderAsc);
    ////printArray(outputDataParallel, dataLen);

    //outputDataCorrect = sortCorrect(inputData, dataLen);
    //compareArrays(outputDataParallel, outputDataCorrect, dataLen);

    ////cudaFreeHost(inputData);
    //cudaFreeHost(outputDataParallel);
    ////free(outputDataSequential);
    //free(outputDataCorrect);

    getchar();
    return 0;
}
