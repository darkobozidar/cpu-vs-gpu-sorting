#include <stdio.h>
#include <climits>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../Utils/data_types_common.h"
#include "constants.h"


/*
Adds the padding to table from start index (original table length) to the end of the extended array (which is
the multiple of number of elements processed by one thread block in initial bitonic sort).
*/
template <data_t value>
__global__ void addPaddingKernel(data_t *dataTable, uint_t start, uint_t length)
{
    const uint_t elemsPerThreadBlock = THREADS_PER_PADDING * ELEMS_PER_THREAD_PADDING;
    const uint_t offset = blockIdx.x * elemsPerThreadBlock;
    const uint_t dataBlockLength = offset + elemsPerThreadBlock <= length ? elemsPerThreadBlock : length - offset;
    data_t *paddedTable = dataTable + start + offset;

    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_PADDING)
    {
        paddedTable[tx] = value;
    }
}

template __global__ void addPaddingKernel<MIN_VAL>(data_t *dataTable, uint_t start, uint_t length);
template __global__ void addPaddingKernel<MAX_VAL>(data_t *dataTable, uint_t start, uint_t length);


/*
Compares 2 elements and exchanges them according to sortOrder.
*/
template <order_t sortOrder>
__device__ void compareExchange(data_t *elem1, data_t *elem2)
{
    if ((*elem1 > *elem2) ^ sortOrder)
    {
        data_t temp = *elem1;
        *elem1 = *elem2;
        *elem2 = temp;
    }
}

/*
Reads the data from global memory to shared memory, sorts it with NORMALIZED bitonic sort and stores the data
back to global memory.
*/
template <order_t sortOrder>
__device__ void bitonicSort(data_t *dataTable, uint_t tableLen)
{
    extern __shared__ data_t bitonicSortTile[];

    const uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    const uint_t offset = blockIdx.x * elemsPerThreadBlock;
    const uint_t dataBlockLength = offset + elemsPerThreadBlock <= tableLen ? elemsPerThreadBlock : tableLen - offset;

    // Reads data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_BITONIC_SORT)
    {
        bitonicSortTile[tx] = dataTable[offset + tx];
    }
    __syncthreads();

    // Bitonic sort PHASES
    for (uint_t subBlockSize = 1; subBlockSize < dataBlockLength; subBlockSize <<= 1)
    {
        // Bitonic merge STEPS
        for (uint_t stride = subBlockSize; stride > 0; stride >>= 1)
        {
            // Each thread processes multiple elements
            for (uint_t tx = threadIdx.x; tx < dataBlockLength >> 1; tx += THREADS_PER_BITONIC_SORT)
            {
                uint_t indexThread = tx;
                uint_t offset = stride;

                // In normalized bitonic sort, first STEP of every PHASE uses different offset than all other STEPS.
                if (stride == subBlockSize)
                {
                    indexThread = (tx / stride) * stride + ((stride - 1) - (tx % stride));
                    offset = ((tx & (stride - 1)) << 1) + 1;
                }

                uint_t index = (indexThread << 1) - (indexThread & (stride - 1));
                if (index + offset >= dataBlockLength)
                {
                    break;
                }

                compareExchange<sortOrder>(&bitonicSortTile[index], &bitonicSortTile[index + offset]);
            }

            __syncthreads();
        }
    }

    // Stores data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_BITONIC_SORT)
    {
        dataTable[offset + tx] = bitonicSortTile[tx];
    }
}

/*
Sorts sub-blocks of input data with NORMALIZED bitonic sort, collects samples and stores them to array for
local samples.
*/
template <order_t sortOrder>
__global__ void bitonicSortCollectSamplesKernel(data_t *dataTable, data_t *localSamples, uint_t tableLen)
{
    extern __shared__ data_t bitonicSortTile[];

    bitonicSort<sortOrder>(dataTable, tableLen);

    // After sort has been performed, samples are collected and stored to array of local samples.
    // Because array is always padded to next multiple of "elemsPerThreadBlock", we can always collect
    // NUM_SAMPLES_PARALLEL from data block.
    const uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    const uint_t localSamplesDistance = elemsPerThreadBlock / NUM_SAMPLES_PARALLEL;
    const uint_t offsetSamples = blockIdx.x * NUM_SAMPLES_PARALLEL;

    // Collects the samples on offset "localSampleDistance / 2" in order to collect them as evenly as possible
    for (uint_t tx = threadIdx.x; tx < NUM_SAMPLES_PARALLEL; tx += THREADS_PER_BITONIC_SORT)
    {
        localSamples[offsetSamples + tx] = bitonicSortTile[tx * localSamplesDistance + (localSamplesDistance / 2)];
    }
}

template __global__ void bitonicSortCollectSamplesKernel<ORDER_ASC>(
    data_t *dataTable, data_t *localSamples, uint_t tableLen
);
template __global__ void bitonicSortCollectSamplesKernel<ORDER_DESC>(
    data_t *dataTable, data_t *localSamples, uint_t tableLen
);


/*
Sorts sub-blocks of input data with NORMALIZED bitonic sort.
*/
template <order_t sortOrder>
__global__ void bitonicSortKernel(data_t *dataTable, uint_t tableLen)
{
    bitonicSort<sortOrder>(dataTable, tableLen);
}

template __global__ void bitonicSortKernel<ORDER_ASC>(data_t *dataTable, uint_t tableLen);
template __global__ void bitonicSortKernel<ORDER_DESC>(data_t *dataTable, uint_t tableLen);


/*
Global bitonic merge for sections, where stride IS GREATER than max shared memory.
*/
template <order_t sortOrder, bool isFirstStepOfPhase>
__global__ void bitonicMergeGlobalKernel(data_t *dataTable, uint_t tableLen, uint_t step)
{
    const uint_t stride = 1 << (step - 1);
    const uint_t pairsPerThreadBlock = (THREADS_PER_GLOBAL_MERGE * ELEMS_PER_THREAD_GLOBAL_MERGE) >> 1;
    const uint_t indexGlobal = blockIdx.x * pairsPerThreadBlock + threadIdx.x;

    for (uint_t i = 0; i < ELEMS_PER_THREAD_GLOBAL_MERGE >> 1; i++)
    {
        uint_t indexThread = indexGlobal + i * THREADS_PER_GLOBAL_MERGE;
        uint_t offset = stride;

        // In normalized bitonic sort, first STEP of every PHASE uses different offset than all other STEPS.
        if (isFirstStepOfPhase)
        {
            offset = ((indexThread & (stride - 1)) << 1) + 1;
            indexThread = (indexThread / stride) * stride + ((stride - 1) - (indexThread % stride));
        }

        uint_t index = (indexThread << 1) - (indexThread & (stride - 1));
        if (index + offset >= tableLen)
        {
            break;
        }

        compareExchange<sortOrder>(&dataTable[index], &dataTable[index + offset]);
    }
}

template __global__ void bitonicMergeGlobalKernel<ORDER_ASC, true>(data_t *dataTable, uint_t tableLen, uint_t step);
template __global__ void bitonicMergeGlobalKernel<ORDER_ASC, false>(data_t *dataTable, uint_t tableLen, uint_t step);
template __global__ void bitonicMergeGlobalKernel<ORDER_DESC, true>(data_t *dataTable, uint_t tableLen, uint_t step);
template __global__ void bitonicMergeGlobalKernel<ORDER_DESC, false>(data_t *dataTable, uint_t tableLen, uint_t step);


/*
Global bitonic merge for sections, where stride IS LOWER OR EQUAL than max shared memory.
*/
template <order_t sortOrder, bool isFirstStepOfPhase>
__global__ void bitonicMergeLocalKernel(data_t *dataTable, uint_t tableLen, uint_t step)
{
    extern __shared__ data_t mergeTile[];
    bool isFirstStepOfPhaseCopy = isFirstStepOfPhase;  // isFirstStepOfPhase is not editable (constant)

    const uint_t elemsPerThreadBlock = THREADS_PER_LOCAL_MERGE * ELEMS_PER_THREAD_LOCAL_MERGE;
    const uint_t offset = blockIdx.x * elemsPerThreadBlock;
    const uint_t dataBlockLength = offset + elemsPerThreadBlock <= tableLen ? elemsPerThreadBlock : tableLen - offset;
    const uint_t pairsPerBlockLength = dataBlockLength >> 1;

    // Reads data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_LOCAL_MERGE)
    {
        mergeTile[tx] = dataTable[offset + tx];
    }
    __syncthreads();

    // Bitonic merge
    for (uint_t stride = 1 << (step - 1); stride > 0; stride >>= 1)
    {
        for (uint_t tx = threadIdx.x; tx < pairsPerBlockLength; tx += THREADS_PER_LOCAL_MERGE)
        {
            uint_t indexThread = tx;
            uint_t offset = stride;

            // In normalized bitonic sort, first STEP of every PHASE uses different offset than all other STEPS.
            if (isFirstStepOfPhaseCopy)
            {
                offset = ((tx & (stride - 1)) << 1) + 1;
                indexThread = (tx / stride) * stride + ((stride - 1) - (tx % stride));
                isFirstStepOfPhaseCopy = false;
            }

            uint_t index = (indexThread << 1) - (indexThread & (stride - 1));
            if (index + offset >= dataBlockLength)
            {
                break;
            }

            compareExchange<sortOrder>(&mergeTile[index], &mergeTile[index + offset]);
        }
        __syncthreads();
    }

    // Stores data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_LOCAL_MERGE)
    {
        dataTable[offset + tx] = mergeTile[tx];
    }
}

template __global__ void bitonicMergeLocalKernel<ORDER_ASC, true>(data_t *dataTable, uint_t tableLen, uint_t step);
template __global__ void bitonicMergeLocalKernel<ORDER_ASC, false>(data_t *dataTable, uint_t tableLen, uint_t step);
template __global__ void bitonicMergeLocalKernel<ORDER_DESC, true>(data_t *dataTable, uint_t tableLen, uint_t step);
template __global__ void bitonicMergeLocalKernel<ORDER_DESC, false>(data_t *dataTable, uint_t tableLen, uint_t step);


/*
From sorted LOCAL samples extracts GLOBAL samples (every NUM_SAMPLES_PARALLEL sample). This is done by one
thread block.
*/
__global__ void collectGlobalSamplesKernel(data_t *samplesLocal, data_t *samplesGlobal, uint_t samplesLen)
{
    const uint_t samplesDistance = samplesLen / NUM_SAMPLES_PARALLEL;

    // Samples are collected on offset (samplesDistance / 2) in order to collect samples as evenly as possible
    samplesGlobal[threadIdx.x] = samplesLocal[threadIdx.x * samplesDistance + (samplesDistance / 2)];
}

/*
Performs a binary INCLUSIVE search and returns index on which element was found.
*/
template <order_t sortOrder>
__device__ int binarySearchInclusive(data_t* dataTable, data_t target, int_t indexStart, int_t indexEnd)
{
    while (indexStart <= indexEnd)
    {
        int index = (indexStart + indexEnd) / 2;

        if (sortOrder == ORDER_ASC ? (target <= dataTable[index]) : (target >= dataTable[index]))
        {
            indexEnd = index - 1;
        }
        else
        {
            indexStart = index + 1;
        }
    }

    return indexStart;
}

/*
In all previously sorted (by initial bitonic sort) sub-blocks finds the indexes of all NUM_SAMPLES_PARALLEL
global samples. From these indexes calculates the number of elements in each of the (NUM_SAMPLES_PARALLEL + 1)
local buckets (calculates local bucket sizes) for every sorted sub-block.

One thread block can process multiple sorted sub-blocks (previously sorted by initial bitonic sort).
Every sub-block is processed by NUM_SAMPLES_PARALLEL threads - every thread block processes
"THREADS_PER_SAMPLE_INDEXING / NUM_SAMPLES_PARALLEL" sorted sub-blocks.
*/
template <order_t sortOrder>
__global__ void sampleIndexingKernel(
    data_t *dataTable, const data_t* __restrict__ samplesGlobal, uint_t *localBucketSizes, uint_t tableLen
)
{
    // Holds indexes of global samples in corresponding sorted (by initial bitonic sort) sub-blocks
    __shared__ uint_t indexingTile[THREADS_PER_SAMPLE_INDEXING];

    const uint_t elemsPerInitBitonicSort = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    const uint_t numAllSubBlocks = tableLen / elemsPerInitBitonicSort;
    const uint_t indexSubBlock = (
        blockIdx.x * THREADS_PER_SAMPLE_INDEXING / NUM_SAMPLES_PARALLEL + threadIdx.x / NUM_SAMPLES_PARALLEL
    );

    const uint_t offset = indexSubBlock * elemsPerInitBitonicSort;
    const uint_t sampleIndex = threadIdx.x % NUM_SAMPLES_PARALLEL;

    // Because one thread block can process multiple sub-blocks previously sorted by initial bitonic sort, sub-block
    // index has to be verified that it is still within array length limit
    if (indexSubBlock < numAllSubBlocks)
    {
        // Searches for global sample index inside sorted sub-block
        indexingTile[threadIdx.x] = binarySearchInclusive<sortOrder>(
            dataTable, samplesGlobal[sampleIndex], offset, offset + elemsPerInitBitonicSort - 1
        );
    }
    __syncthreads();

    const uint_t outputBucketIndex = sampleIndex * numAllSubBlocks + indexSubBlock;
    const uint_t prevIndex = sampleIndex == 0 ? offset : indexingTile[threadIdx.x - 1];

    // From global sample indexes calculates and stores bucket sizes
    if (indexSubBlock < numAllSubBlocks)
    {
        localBucketSizes[outputBucketIndex] = indexingTile[threadIdx.x] - prevIndex;

        // Because there is NUM_SAMPLES_PARALLEL samples, (NUM_SAMPLES_PARALLEL + 1) buckets are created. Last
        // thread assigned to every sub-block calculates and stores bucket size for (NUM_SAMPLES_PARALLEL + 1)
        // bucket
        if (sampleIndex == NUM_SAMPLES_PARALLEL - 1)
        {
            uint_t bucketSize = offset + elemsPerInitBitonicSort - indexingTile[threadIdx.x];
            localBucketSizes[outputBucketIndex + numAllSubBlocks] = bucketSize;
        }
    }
}

template __global__ void sampleIndexingKernel<ORDER_ASC>(
    data_t *dataTable, const data_t* __restrict__ samples, uint_t * bucketSizes, uint_t tableLen
);
template __global__ void sampleIndexingKernel<ORDER_DESC>(
    data_t *dataTable, const data_t* __restrict__ samples, uint_t * bucketSizes, uint_t tableLen
);

/*
According to local (per one sorted sub-block) bucket sizes and offsets, kernel scatters elements to their global
buckets. Last thread block also calculates global bucket offsets (from local bucket sizes and offsets) and stores
them.
*/
__global__ void bucketsRelocationKernel(
    data_t *dataTable, data_t *dataBuffer, uint_t *globalBucketOffsets, const uint_t* __restrict__ localBucketSizes,
    const uint_t* __restrict__ localBucketOffsets, uint_t tableLen
)
{
    extern __shared__ uint_t bucketsTile[];
    uint_t *bucketSizes = bucketsTile;
    uint_t *bucketOffsets = bucketsTile + NUM_SAMPLES_PARALLEL + 1;

    // Reads bucket sizes and offsets to shared memory
    if (threadIdx.x < NUM_SAMPLES_PARALLEL + 1)
    {
        uint_t index = threadIdx.x * gridDim.x + blockIdx.x;
        bucketSizes[threadIdx.x] = localBucketSizes[index];
        bucketOffsets[threadIdx.x] = localBucketOffsets[index];

        // Last thread block writes size of entire buckets into array of global bucket sizes
        if (blockIdx.x == gridDim.x - 1)
        {
            globalBucketOffsets[threadIdx.x] = bucketOffsets[threadIdx.x] + bucketSizes[threadIdx.x];
        }

        // If thread block contains only NUM_SAMPLES_PARALLEL threads (which is min number of threads per this
        // kernel), then last thread also reads (NUM_SAMPLES_PARALLEL + 1)th bucket
        if (THREADS_PER_BUCKETS_RELOCATION == NUM_SAMPLES_PARALLEL && threadIdx.x == NUM_SAMPLES_PARALLEL - 1)
        {
            bucketSizes[threadIdx.x + 1] = localBucketSizes[index + gridDim.x];
            bucketOffsets[threadIdx.x + 1] = localBucketOffsets[index + gridDim.x];

            if (blockIdx.x == gridDim.x - 1)
            {
                globalBucketOffsets[threadIdx.x + 1] = bucketOffsets[threadIdx.x + 1] + bucketSizes[threadIdx.x + 1];
            }
        }
    }
    __syncthreads();

    const uint_t elemsPerBitonicSort = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    const uint_t offset = blockIdx.x * elemsPerBitonicSort;
    const uint_t dataBlockLength = offset + elemsPerBitonicSort <= tableLen ? elemsPerBitonicSort : tableLen - offset;
    uint_t activeThreads = 0;
    uint_t activeThreadsPrev = 0;

    uint_t tx = threadIdx.x;
    uint_t bucketIndex = 0;

    // Every thread reads bucket size and scatters it's corresponding elements to their global buckets
    while (tx < dataBlockLength)
    {
        activeThreads += bucketSizes[bucketIndex];

        // Stores elements to it's corresponding bucket until bucket is filled
        while (tx < activeThreads)
        {
            dataBuffer[bucketOffsets[bucketIndex] + tx - activeThreadsPrev] = dataTable[offset + tx];
            tx += THREADS_PER_BUCKETS_RELOCATION;
        }

        // When bucket is filled, moves to next bucket
        activeThreadsPrev = activeThreads;
        bucketIndex++;
    }
}
