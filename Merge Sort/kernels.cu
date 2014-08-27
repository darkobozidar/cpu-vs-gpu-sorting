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
	// Thread index for first or second half of the sub-table
	uint_t threadIdxX = threadIdx.x + (!firstHalf) * blockDim.x;
	uint_t subBlocksPerBlock = tabBlockSize / tabSubBlockSize;
	// Index of a block from which thread will read the sample
	uint_t indexBlock = threadIdxX / subBlocksPerBlock;
	// Offset to block index (we devide and multiply with same value, to lose the offset to sub-block)
	uint_t index = indexBlock * subBlocksPerBlock;
	// Offset for sub-block index inside block for ODD block
	index += ((indexBlock % 2 == 0) * threadIdxX) % subBlocksPerBlock;
	// Offset for sub-block index inside block for EVEN block (index has to be reversed)
	index += ((indexBlock % 2 == 1) * (subBlocksPerBlock - (threadIdxX + 1))) % subBlocksPerBlock;

	return index;
}

__device__ void printfOnce(char* text) {
	if (threadIdx.x == 0) {
		printf(text);
	}
}

__global__ void generateSublocksKernel(data_t* table, uint_t tableLen, uint_t tabBlockSize, uint_t tabSubBlockSize) {
	extern __shared__ data_t tile[];
	uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x * tabSubBlockSize;
	uint_t sharedMemIdx1, sharedMemIdx2;
	data_t value1, value2;

	// Values are read in coalesced way...
	if (index < tableLen) {
		value1 = table[index];
	}
	if (index + blockDim.x < tableLen) {
		value2 = table[index + blockDim.x * tabSubBlockSize];
	}

	// ...and than reversed when added to shared memory
	sharedMemIdx1 = calculateElementIndex(tableLen, tabBlockSize, tabSubBlockSize, true);
	sharedMemIdx2 = calculateElementIndex(tableLen, tabBlockSize, tabSubBlockSize, false);
	tile[sharedMemIdx1] = value1;
	tile[sharedMemIdx2] = value2;

	for (uint_t stride = tabBlockSize / tabSubBlockSize; stride > 0; stride /= 2) {
		__syncthreads();
		uint_t index = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

		// TODO use max/min or conditional operator (or something else)
		if (tile[index] > tile[index + stride]) {
			data_t temp = tile[index];
			tile[index] = tile[index + stride];
			tile[index + stride] = temp;
		}
	}
}
