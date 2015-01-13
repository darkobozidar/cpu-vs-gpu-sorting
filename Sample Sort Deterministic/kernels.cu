#include <stdio.h>
#include <climits>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../Utils/data_types_common.h"
#include "constants.h"


/*
Adds the padding to table from start index (original table length, which is not power of 2) to the end of the
extended array (which is the next power of 2 of the original table length). Needed because of bitonic sort, for
which table length divisable by 2 is needed.
*/
template <data_t value>
__global__ void addPaddingKernel(data_t *dataTable, uint_t start, uint_t length)
{
    uint_t elemsPerThreadBlock = THREADS_PER_PADDING * ELEMS_PER_THREAD_PADDING;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    uint_t dataBlockLength = offset + elemsPerThreadBlock <= length ? elemsPerThreadBlock : length - offset;
    offset += start;

    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_PADDING)
    {
        dataTable[offset + tx] = value;
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

template <order_t sortOrder>
__device__ void bitonicSort(data_t *dataTable, uint_t tableLen)
{
    extern __shared__ data_t bitonicSortTile[];

    const uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    const uint_t offset = blockIdx.x * elemsPerThreadBlock;
    const uint_t dataBlockLength = offset + elemsPerThreadBlock <= tableLen ? elemsPerThreadBlock : tableLen - offset;

    // Read data from global to shared memory.
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

    // Store data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_BITONIC_SORT)
    {
        dataTable[offset + tx] = bitonicSortTile[tx];
    }
}

/*
Sorts sub-blocks of input data with NORMALIZED bitonic sort and collects samples in array for local samples.
*/
template <order_t sortOrder>
__global__ void bitonicSortCollectSamplesKernel(data_t *dataTable, data_t *localSamples, uint_t tableLen)
{
    extern __shared__ data_t bitonicSortTile[];

    bitonicSort<sortOrder>(dataTable, tableLen);

    // After sort has been performed, samples are scattered to array of local samples
    const uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    const uint_t offset = blockIdx.x * elemsPerThreadBlock;
    const uint_t dataBlockLength = offset + elemsPerThreadBlock <= tableLen ? elemsPerThreadBlock : tableLen - offset;

    const uint_t localSamplesDistance = elemsPerThreadBlock / NUM_SAMPLES;
    const uint_t samplesPerThreadBlock = (dataBlockLength - 1) / localSamplesDistance + 1;

    // Collects samples
    for (uint_t tx = threadIdx.x; tx < samplesPerThreadBlock; tx += THREADS_PER_BITONIC_SORT)
    {
        // Collects the samples on offset of "localSampleDistance / 2" in order to be nearer to center
        data_t sample = bitonicSortTile[localSamplesDistance / 2 + tx * localSamplesDistance];
        localSamples[blockIdx.x * NUM_SAMPLES + tx] = sample;
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
From LOCAL samples extracts GLOBAL samples (every NUM_SAMPLES sample). This is done by one thread block.
*/
__global__ void collectGlobalSamplesKernel(data_t *samples, uint_t samplesLen)
{
    // Shared memory is needed, because samples are read and written to the same array (race condition).
    __shared__ data_t globalSamplesTile[NUM_SAMPLES];
    uint_t samplesDistance = samplesLen / NUM_SAMPLES;

    // We also add (samplesDistance / 2) in order to collect samples as evenly as possible
    globalSamplesTile[threadIdx.x] = samples[threadIdx.x * samplesDistance + (samplesDistance / 2)];
    __syncthreads();
    samples[threadIdx.x] = globalSamplesTile[threadIdx.x];
}

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
For all previously sorted chunks finds the index of global samples and calculates the number of elements in each
of the (NUM_SAMPLES + 1) buckets.
*/
template <order_t sortOrder>
__global__ void sampleIndexingKernel(
    data_t *dataTable, const data_t* __restrict__ samples, uint_t * bucketSizes, uint_t tableLen
)
{
    __shared__ uint_t indexingTile[THREADS_PER_SAMPLE_INDEXING];

    uint_t sampleIndex = threadIdx.x % NUM_SAMPLES;

    // One thread block can process multiple data blocks (multiple chunks of data previously sorted by bitonic sort).
    const uint_t dataBlocksPerThreadBlock = (blockDim.x - 1) / NUM_SAMPLES + 1;
    const uint_t dataBlockIndex = threadIdx.x / NUM_SAMPLES;
    const uint_t elemsPerBitonicSort = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;

    const uint_t indexBlock = (blockIdx.x * dataBlocksPerThreadBlock + dataBlockIndex);
    const uint_t offset = indexBlock * elemsPerBitonicSort;
    const uint_t dataBlockLength = offset + elemsPerBitonicSort <= tableLen ? elemsPerBitonicSort : tableLen - offset;

    indexingTile[threadIdx.x] = binarySearchInclusive<sortOrder>(
        dataTable, samples[sampleIndex], offset, offset + dataBlockLength - 1
    );
    __syncthreads();

    uint_t prevIndex;
    const uint_t allDataBlocks = gridDim.x * dataBlocksPerThreadBlock;
    const uint_t outputSampleIndex = sampleIndex * allDataBlocks + indexBlock;

    if (sampleIndex == 0)
    {
        prevIndex = offset;
    }
    else
    {
        prevIndex = indexingTile[threadIdx.x - 1];
    }
    __syncthreads();

    bucketSizes[outputSampleIndex] = indexingTile[threadIdx.x] - prevIndex;
    // Because there is NUM_SAMPLES samples, (NUM_SAMPLES + 1) buckets are created.
    if (sampleIndex == NUM_SAMPLES - 1)
    {
        bucketSizes[outputSampleIndex + allDataBlocks] = offset + dataBlockLength - indexingTile[threadIdx.x];
    }
}

template __global__ void sampleIndexingKernel<ORDER_ASC>(
    data_t *dataTable, const data_t* __restrict__ samples, uint_t * bucketSizes, uint_t tableLen
);
template __global__ void sampleIndexingKernel<ORDER_DESC>(
    data_t *dataTable, const data_t* __restrict__ samples, uint_t * bucketSizes, uint_t tableLen
);

/*
According to local (per one tile) bucket sizes and offsets, kernel scatters elements to their global buckets.
*/
__global__ void bucketsRelocationKernel(
    data_t*dataTable, data_t *dataBuffer, uint_t *d_globalBucketOffsets, const uint_t* __restrict__ localBucketSizes,
    const uint_t* __restrict__ localBucketOffsets, uint_t tableLen
)
{
    extern __shared__ uint_t bucketsTile[];
    uint_t *bucketSizes = bucketsTile;
    uint_t *bucketOffsets = bucketsTile + NUM_SAMPLES + 1;

    // Reads bucket sizes and offsets to shared memory
    if (threadIdx.x < NUM_SAMPLES + 1)
    {
        uint_t index = threadIdx.x * gridDim.x + blockIdx.x;
        bucketSizes[threadIdx.x] = localBucketSizes[index];
        bucketOffsets[threadIdx.x] = localBucketOffsets[index];

        // Last block writes size of entire buckets into array of global bucket sizes
        if (blockIdx.x == gridDim.x - 1)
        {
            d_globalBucketOffsets[threadIdx.x] = bucketOffsets[threadIdx.x] + bucketSizes[threadIdx.x];
        }

        // If thread block contains only NUM_SAMPLES threads (which is min number of threads per this kernel),
        // then last thread reads also (NUM_SAMPLES + 1)th bucket
        if (THREADS_PER_BUCKETS_RELOCATION == NUM_SAMPLES && threadIdx.x == NUM_SAMPLES - 1)
        {
            bucketSizes[threadIdx.x + 1] = localBucketSizes[index + gridDim.x];
            bucketOffsets[threadIdx.x + 1] = localBucketOffsets[index + gridDim.x];

            if (blockIdx.x == gridDim.x - 1)
            {
                d_globalBucketOffsets[threadIdx.x + 1] = bucketOffsets[threadIdx.x + 1] + bucketSizes[threadIdx.x + 1];
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

    // Every thread reads bucket size and scatters elements to their global buckets
    while (tx < dataBlockLength)
    {
        activeThreads += bucketSizes[bucketIndex];

        while (tx < activeThreads)
        {
            dataBuffer[bucketOffsets[bucketIndex] + tx - activeThreadsPrev] = dataTable[offset + tx];
            tx += THREADS_PER_BUCKETS_RELOCATION;
        }

        activeThreadsPrev = activeThreads;
        bucketIndex++;
    }
}
