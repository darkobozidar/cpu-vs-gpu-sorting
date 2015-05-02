#ifndef KERNELS_COMMON_SAMPLE_SORT_H
#define KERNELS_COMMON_SAMPLE_SORT_H

#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../../Utils/data_types_common.h"
#include "../../Utils/kernels_utils.h"


/*
From sorted LOCAL samples extracts GLOBAL samples (every numSamples sample). This is done by one thread block.
*/
template <uint_t numSamples>
__global__ void collectGlobalSamplesKernel(data_t *samplesLocal, data_t *samplesGlobal, uint_t samplesLen)
{
    const uint_t samplesDistance = samplesLen / numSamples;

    // Samples are collected on offset (samplesDistance / 2) in order to collect samples as evenly as possible
    samplesGlobal[threadIdx.x] = samplesLocal[threadIdx.x * samplesDistance + (samplesDistance / 2)];
}

/*
In all previously sorted (by initial bitonic sort) sub-blocks finds the indexes of all numSamples global samples.
From these indexes calculates the number of elements in each of the (numSamples + 1) local buckets (calculates
local bucket sizes) for every sorted sub-block.

One thread block can process multiple sorted sub-blocks (previously sorted by initial bitonic sort). Every sub-block
is processed by numSamples threads - every thread block processes "threadsSampleIndexing / numSamples" sorted
sub-blocks.
*/
template <
    uint_t threadsSampleIndexing, uint_t threadsBitonicSort, uint_t elemsBitonicSort, uint_t numSamples,
    order_t sortOrder
>
__global__ void sampleIndexingKernel(
    data_t *dataTable, const data_t* __restrict__ samplesGlobal, uint_t *localBucketSizes, uint_t tableLen
)
{
    // Holds indexes of global samples in corresponding sorted (by initial bitonic sort) sub-blocks
    __shared__ uint_t indexingTile[threadsSampleIndexing];

    const uint_t elemsPerInitBitonicSort = threadsBitonicSort * elemsBitonicSort;
    const uint_t numAllSubBlocks = tableLen / elemsPerInitBitonicSort;
    const uint_t indexSubBlock = blockIdx.x * threadsSampleIndexing / numSamples + threadIdx.x / numSamples;

    const uint_t offset = indexSubBlock * elemsPerInitBitonicSort;
    const uint_t sampleIndex = threadIdx.x % numSamples;

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

        // Because there is numSamples samples, (numSamples + 1) buckets are created. Last thread assigned to
        // every sub-block calculates and stores bucket size for (numSamples + 1) bucket
        if (sampleIndex == numSamples - 1)
        {
            uint_t bucketSize = offset + elemsPerInitBitonicSort - indexingTile[threadIdx.x];
            localBucketSizes[outputBucketIndex + numAllSubBlocks] = bucketSize;
        }
    }
}

/*
According to local (per one sorted sub-block) bucket sizes and offsets, kernel scatters elements to their global
buckets. Last thread block also calculates global bucket offsets (from local bucket sizes and offsets) and stores
them.
*/
template <
    uint_t threadsBucketsRelocation, uint_t threadsBitonicSort, uint_t elemsBitonicSort, uint_t numSamples,
    bool sortingKeyOnly
>
__global__ void bucketsRelocationKernel(
    data_t *dataKeys, data_t *dataValues, data_t *bufferKeys, data_t *bufferValues, uint_t *globalBucketOffsets,
    const uint_t* __restrict__ localBucketSizes, const uint_t* __restrict__ localBucketOffsets, uint_t tableLen
)
{
    extern __shared__ uint_t bucketsTile[];
    uint_t *bucketSizes = bucketsTile;
    uint_t *bucketOffsets = bucketsTile + numSamples + 1;

    // Reads bucket sizes and offsets to shared memory
    if (threadIdx.x < numSamples + 1)
    {
        uint_t index = threadIdx.x * gridDim.x + blockIdx.x;
        bucketSizes[threadIdx.x] = localBucketSizes[index];
        bucketOffsets[threadIdx.x] = localBucketOffsets[index];

        // Last thread block writes size of entire buckets into array of global bucket sizes
        if (blockIdx.x == gridDim.x - 1)
        {
            globalBucketOffsets[threadIdx.x] = bucketOffsets[threadIdx.x] + bucketSizes[threadIdx.x];
        }

        // If thread block contains only numSamples threads (which is min number of threads per this
        // kernel), then last thread also reads (numSamples + 1)th bucket
        if (threadsBucketsRelocation == numSamples && threadIdx.x == numSamples - 1)
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

    uint_t activeThreads = 0;
    uint_t activeThreadsPrev = 0;
    uint_t offset, dataBlockLength;
    calcDataBlockLength<threadsBitonicSort, elemsBitonicSort>(offset, dataBlockLength, tableLen);

    uint_t tx = threadIdx.x;
    uint_t bucketIndex = 0;

    // Every thread reads bucket size and scatters it's corresponding elements to their global buckets
    while (tx < dataBlockLength)
    {
        activeThreads += bucketSizes[bucketIndex];

        // Stores elements to it's corresponding bucket until bucket is filled
        for (; tx < activeThreads; tx += threadsBucketsRelocation)
        {
            bufferKeys[bucketOffsets[bucketIndex] + tx - activeThreadsPrev] = dataKeys[offset + tx];
            if (!sortingKeyOnly)
            {
                bufferValues[bucketOffsets[bucketIndex] + tx - activeThreadsPrev] = dataValues[offset + tx];
            }
        }

        // When bucket is filled, moves to next bucket
        activeThreadsPrev = activeThreads;
        bucketIndex++;
    }
}

#endif
