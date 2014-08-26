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


int comparator(const void * elem1, const void * elem2) {
	return (*(data_t*)elem1 - *(data_t*)elem2);
}

int main(int argc, char** argv) {
	data_t* input;
	data_t* outputParallel;
	data_t* outputSequential;
	data_t* correctlySorted;

	uint_t arrayLen = 1 << 5;
	uint_t blockSize;
	bool orderAsc = TRUE;
	cudaError_t error;

	LARGE_INTEGER timerStart;

	cudaFree(NULL);  // Initializes CUDA, because CUDA init is lazy
	srand(time(NULL));

	error = cudaHostAlloc(&input, arrayLen * sizeof(*input), cudaHostAllocDefault);
	checkCudaError(error);
	error = cudaHostAlloc(&outputParallel, arrayLen * sizeof(*outputParallel), cudaHostAllocDefault);
	checkCudaError(error);
	fillArrayRand(input, arrayLen);

	blockSize = sortParallel(input, outputParallel, arrayLen, orderAsc);

	startStopwatch(&timerStart);
	correctlySorted = copyArray(input, arrayLen);  // TODO Use some C implementation of copy array
	for (int i = 0; i < arrayLen / blockSize; i++) {
		qsort(correctlySorted + i * blockSize, blockSize, sizeof(*correctlySorted), comparator);
	}
	endStopwatch(timerStart, "Sequential sort");

	printf("\n\n");
	compareArrays(outputParallel, correctlySorted, arrayLen);

	printArray(outputParallel, arrayLen);

	getchar();
	return 0;
}
