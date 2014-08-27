#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "data_types.h"
#include "constants.h"


__global__ void bitonicSortKernel(data_t* array, uint_t arrayLen, uint_t sharedMemSize) {
	extern __shared__ data_t tile[];
	uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;
	uint_t numStages = ceil(log2((double) sharedMemSize));

	if (index < arrayLen) {
		tile[threadIdx.x] = array[index];
	}
	if (index + blockDim.x < arrayLen) {
		tile[threadIdx.x + blockDim.x] = array[index + blockDim.x];
	}

	for (uint_t stage = 0; stage < numStages; stage++) {
		for (uint_t pass = 0; pass <= stage; pass++) {
			__syncthreads();

			uint_t pairDistance = 1 << (stage - pass);
			uint_t blockWidth = 2 * pairDistance;
			uint_t leftId = (threadIdx.x & (pairDistance - 1)) + (threadIdx.x >> (stage - pass)) * blockWidth;
			uint_t rightId = leftId + pairDistance;

			data_t leftElement, rightElement;
			data_t greater, lesser;
			leftElement = tile[leftId];
			rightElement = tile[rightId];

			uint_t sameDirectionBlockWidth = threadIdx.x >> stage;
			uint_t sameDirection = sameDirectionBlockWidth & 0x1;

			uint_t temp = sameDirection ? rightId : temp;
			rightId = sameDirection ? leftId : rightId;
			leftId = sameDirection ? temp : leftId;

			bool compareResult = (leftElement < rightElement);
			greater = compareResult ? rightElement : leftElement;
			lesser = compareResult ? leftElement : rightElement;

			tile[leftId] = lesser;
			tile[rightId] = greater;
		}
	}

	__syncthreads();

	if (index < arrayLen) {
		array[index] = tile[threadIdx.x];
	}
	if (index + blockDim.x < arrayLen) {
		array[index + blockDim.x] = tile[threadIdx.x + blockDim.x];
	}
}

__device__ uint_t calculateElementIndex(uint_t tableLen, uint_t tabBlockSize, uint_t tabSubBlockSize, bool firstHalf) {
	// Thread index for first or second half of the table
	uint_t threadIdxX = threadIdx.x + (!firstHalf) * blockDim.x;
	// Index of a block which thread will read the sample from
	uint_t indexBlock = threadIdxX / (tabBlockSize / tabSubBlockSize);
	// Offset to block index (every thread loads 2 elements)
	uint_t index = blockIdx.x * 2 * blockDim.x + indexBlock * tabBlockSize;
	// Offset for sub-block index inside block for ODD block
	index += ((indexBlock % 2 == 0) * threadIdxX * tabSubBlockSize) % tabBlockSize;
	// Offset for sub-block index inside block for EVEN block (index has to be reversed)
	index += ((indexBlock % 2 == 1) * (tabBlockSize - (threadIdxX + 1) * tabSubBlockSize)) % tabBlockSize;

	return index;
}

__global__ void generateSublocksKernel(data_t* table, uint_t tableLen, uint_t tabBlockSize, uint_t tabSubBlockSize) {
	extern __shared__ data_t tile[];
	uint_t index1 = calculateElementIndex(tableLen, tabBlockSize, tabSubBlockSize, true);
	uint_t index2 = calculateElementIndex(tableLen, tabBlockSize, tabSubBlockSize, false);

	if (index1 < tableLen) {
		tile[threadIdx.x] = table[index1];
	}
	if (index2 + tableLen / 2 < tableLen) {
		tile[threadIdx.x + blockDim.x] = table[index2];
	}

	for (uint_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
		__syncthreads();
		uint_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

		if (tile[pos] > tile[pos + stride]) {
			data_t temp = tile[pos];
			tile[pos] = tile[pos + stride];
			tile[pos + stride] = temp;
		}
	}
}
