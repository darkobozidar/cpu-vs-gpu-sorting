#include <stdio.h>
#include <math.h>
#include <Windows.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cudpp.h>

#include "../Utils/data_types_common.h"
#include "../Utils/cuda.h"
#include "../Utils/host.h"
#include "constants.h"
#include "kernels.h"


/*
Initializes CUDPP scan.
*/
void cudppInitScan(CUDPPHandle *scanPlan, uint_t tableLen)
{
    // Initializes the CUDPP Library
    CUDPPHandle theCudpp;
    cudppCreate(&theCudpp);

    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_UINT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;

    *scanPlan = 0;
    CUDPPResult result = cudppPlan(theCudpp, scanPlan, config, tableLen, 1, 0);

    if (result != CUDPP_SUCCESS) {
        printf("Error creating CUDPPPlan\n");
        getchar();
        exit(-1);
    }
}

/*
Sorts sub-blocks of input data with bitonic sort and collects samples after the sort is complete.
*/
void runBitonicSortCollectSamplesKernel(data_t *dataTable, data_t *samples, uint_t tableLen, order_t sortOrder)
{
    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    uint_t sharedMemSize = elemsPerThreadBlock * sizeof(*dataTable);

    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_BITONIC_SORT, 1, 1);

    if (sortOrder == ORDER_ASC)
    {
        bitonicSortCollectSamplesKernel<ORDER_ASC><<<dimGrid, dimBlock, sharedMemSize>>>(
            dataTable, samples, tableLen
        );
    }
    else
    {
        bitonicSortCollectSamplesKernel<ORDER_DESC><<<dimGrid, dimBlock, sharedMemSize>>>(
            dataTable, samples, tableLen
        );
    }
}

/*
Sorts sub-blocks of input data with bitonic sort.
*/
void runBitonicSortKernel(data_t *dataTable, uint_t tableLen, order_t sortOrder)
{
    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    uint_t sharedMemSize = elemsPerThreadBlock * sizeof(*dataTable);

    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_BITONIC_SORT, 1, 1);

    if (sortOrder == ORDER_ASC)
    {
        bitonicSortKernel<ORDER_ASC><<<dimGrid, dimBlock, sharedMemSize>>>(
            dataTable, tableLen
        );
    }
    else
    {
        bitonicSortKernel<ORDER_DESC><<<dimGrid, dimBlock, sharedMemSize>>>(
            dataTable, tableLen
        );
    }
}

void runBitonicMergeGlobalKernel(
    data_t *dataTable, uint_t tableLen, uint_t phase, uint_t step, order_t sortOrder
)
{
    uint_t elemsPerThreadBlock = THREADS_PER_GLOBAL_MERGE * ELEMS_PER_THREAD_GLOBAL_MERGE;
    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_GLOBAL_MERGE, 1, 1);

    bool isFirstStepOfPhase = phase == step;

    if (sortOrder == ORDER_ASC)
    {
        if (isFirstStepOfPhase)
        {
            bitonicMergeGlobalKernel<ORDER_ASC, true><<<dimGrid, dimBlock>>>(dataTable, tableLen, step);
        }
        else
        {
            bitonicMergeGlobalKernel<ORDER_ASC, false><<<dimGrid, dimBlock>>>(dataTable, tableLen, step);
        }
    }
    else
    {
        if (isFirstStepOfPhase)
        {
            bitonicMergeGlobalKernel<ORDER_DESC, true><<<dimGrid, dimBlock>>>(dataTable, tableLen, step);
        }
        else
        {
            bitonicMergeGlobalKernel<ORDER_DESC, false><<<dimGrid, dimBlock>>>(dataTable, tableLen, step);
        }
    }
}

void runBitoicMergeLocalKernel(data_t *dataTable, uint_t tableLen, uint_t phase, uint_t step, order_t sortOrder)
{
    // Every thread loads and sorts 2 elements
    uint_t elemsPerThreadBlock = THREADS_PER_LOCAL_MERGE * ELEMS_PER_THREAD_LOCAL_MERGE;
    uint_t sharedMemSize = elemsPerThreadBlock * sizeof(*dataTable);

    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_LOCAL_MERGE, 1, 1);

    bool isFirstStepOfPhase = phase == step;

    if (sortOrder == ORDER_ASC)
    {
        if (isFirstStepOfPhase)
        {
            bitonicMergeLocalKernel<ORDER_ASC, true><<<dimGrid, dimBlock, sharedMemSize>>>(
                dataTable, tableLen, step
            );
        }
        else
        {
            bitonicMergeLocalKernel<ORDER_ASC, false><<<dimGrid, dimBlock, sharedMemSize>>>(
                dataTable, tableLen, step
            );
        }
    }
    else
    {
        if (isFirstStepOfPhase)
        {
            bitonicMergeLocalKernel<ORDER_DESC, true><<<dimGrid, dimBlock, sharedMemSize>>>(
                dataTable, tableLen, step
            );
        }
        else
        {
            bitonicMergeLocalKernel<ORDER_DESC, false><<<dimGrid, dimBlock, sharedMemSize>>>(
                dataTable, tableLen, step
            );
        }
    }
}

/*
From all LOCAL samples collects (NUM_SAMPLES) GLOBAL samples.
*/
void runCollectGlobalSamplesKernel(data_t *samples, uint_t samplesLen)
{
    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(NUM_SAMPLES, 1, 1);

    collectGlobalSamplesKernel<<<dimGrid, dimBlock>>>(samples, samplesLen);
}

/*
For every sample searches, how many elements in tile are lower than it's value.
*/
void runSampleIndexingKernel(
    data_t *dataTable, data_t *samples, data_t *bucketSizes, uint_t tableLen, uint_t numAllBuckets,
    order_t sortOrder
)
{
    // Number of threads per thread block can be greater than number of samples.
    uint_t elemsPerBitonicSort = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    uint_t numBlocks = (tableLen - 1) / elemsPerBitonicSort + 1;
    uint_t threadBlockSize = min(numBlocks * NUM_SAMPLES, THREADS_PER_SAMPLE_INDEXING);

    // Every thread block creates from NUM_SAMPLES samples (NUM_SAMPLES + 1) buckets
    dim3 dimGrid((numAllBuckets - 1) / (threadBlockSize / NUM_SAMPLES * (NUM_SAMPLES + 1)) + 1, 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    if (sortOrder == ORDER_ASC)
    {
        sampleIndexingKernel<ORDER_ASC><<<dimGrid, dimBlock>>>(dataTable, samples, bucketSizes, tableLen);
    }
    else
    {
        sampleIndexingKernel<ORDER_DESC><<<dimGrid, dimBlock>>>(dataTable, samples, bucketSizes, tableLen);
    }
}

/*
From local bucket sizes and offsets scatters elements to their global buckets. At the end it coppies
global bucket sizes (sizes of whole buckets, not just bucket size per tile (local size)) to host.
*/
void runBucketsRelocationKernel(
    data_t *dataTable, data_t *dataBuffer, uint_t *h_globalBucketOffsets, uint_t *d_globalBucketOffsets,
    uint_t *localBucketSizes, uint_t *localBucketOffsets, uint_t tableLen
)
{
    // For NUM_SAMPLES samples (NUM_SAMPLES + 1) buckets are created
    uint_t sharedMemSize = 2 * (NUM_SAMPLES + 1) * sizeof(*localBucketSizes);
    uint_t elemsPerBitonicSort = THREADS_PER_GLOBAL_MERGE * ELEMS_PER_THREAD_GLOBAL_MERGE;
    cudaError_t error;

    dim3 dimGrid((tableLen - 1) / elemsPerBitonicSort + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_BUCKETS_RELOCATION, 1, 1);

    bucketsRelocationKernel<<<dimGrid, dimBlock, sharedMemSize>>>(
        dataTable, dataBuffer, d_globalBucketOffsets, localBucketSizes, localBucketOffsets, tableLen
    );

    error = cudaMemcpy(
        h_globalBucketOffsets, d_globalBucketOffsets, (NUM_SAMPLES + 1) * sizeof(*h_globalBucketOffsets),
        cudaMemcpyDeviceToHost
    );
    checkCudaError(error);
}

/*
Performs global bitonic merge, when number of elements is greater than shared memory size.
*/
void bitonicMerge(data_t *dataTable, uint_t tableLen, uint_t elemsPerBlockBitonicSort, order_t sortOrder)
{
    uint_t tableLenPower2 = nextPowerOf2(tableLen);
    uint_t elemsPerBlockMergeLocal = THREADS_PER_LOCAL_MERGE * ELEMS_PER_THREAD_LOCAL_MERGE;

    // Number of phases, which can be executed in shared memory (stride is lower than shared memory size)
    uint_t phasesBitonicSort = log2((double)min(tableLenPower2, elemsPerBlockBitonicSort));
    uint_t phasesMergeLocal = log2((double)min(tableLenPower2, elemsPerBlockMergeLocal));
    uint_t phasesAll = log2((double)tableLenPower2);

    for (uint_t phase = phasesBitonicSort + 1; phase <= phasesAll; phase++)
    {
        uint_t step = phase;
        while (step > phasesMergeLocal)
        {
            runBitonicMergeGlobalKernel(dataTable, tableLen, phase, step, sortOrder);
            step--;
        }

        runBitoicMergeLocalKernel(dataTable, tableLen, phase, step, sortOrder);
    }
}

/*
Performs bitonic sort.
*/
void bitonicSort(data_t *dataTable, uint_t tableLen, order_t sortOrder)
{
    uint_t elemsPerBlockBitonicSort = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;

    runBitonicSortKernel(dataTable, tableLen, sortOrder);
    bitonicMerge(dataTable, tableLen, elemsPerBlockBitonicSort, sortOrder);
}

/*
Sorts array with deterministic sample sort.
*/
void sampleSort(
    data_t *&dataTable, data_t *&dataBuffer, data_t *samples, uint_t *h_globalBucketOffsets,
    uint_t *d_globalBucketOffsets, uint_t *d_localBucketSizes, uint_t *d_localBucketOffsets, uint_t tableLen,
    order_t sortOrder
)
{
    uint_t elemsPerInitBitonicSort = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    uint_t localSamplesDistance = (THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT) / NUM_SAMPLES;
    uint_t localSamplesLen = (tableLen - 1) / localSamplesDistance + 1;
    // (number of all data blocks (tiles)) * (number buckets generated from NUM_SAMPLES)
    uint_t localBucketsLen = ((tableLen - 1) / elemsPerInitBitonicSort + 1) * (NUM_SAMPLES + 1);
    CUDPPHandle scanPlan;

    cudppInitScan(&scanPlan, localBucketsLen);
    runBitonicSortCollectSamplesKernel(dataTable, samples, tableLen, sortOrder);

    // Array has already been sorted
    if (tableLen <= elemsPerInitBitonicSort) {
        data_t *temp = dataTable;
        dataTable = dataBuffer;
        dataBuffer = temp;

        return;
    }

    // Local samples are already partially ordered - NUM_SAMPLES per every tile. These partially ordered
    // samples have to be merged.
    bitonicMerge(samples, localSamplesLen, NUM_SAMPLES, sortOrder);

    // TODO handle case, if all samples are the same
    runCollectGlobalSamplesKernel(samples, localSamplesLen);
    runSampleIndexingKernel(dataTable, samples, d_localBucketSizes, tableLen, localBucketsLen, sortOrder);

    CUDPPResult result = cudppScan(scanPlan, d_localBucketOffsets, d_localBucketSizes, localBucketsLen);
    if (result != CUDPP_SUCCESS)
    {
        printf("Error in cudppScan()\n");
        getchar();
        exit(-1);
    }

    runBucketsRelocationKernel(
        dataTable, dataBuffer, h_globalBucketOffsets, d_globalBucketOffsets, d_localBucketSizes,
        d_localBucketOffsets, tableLen
    );

    // Sorts every bucket with bitonic sort
    uint_t previousOffset = 0;
    for (uint_t bucket = 0; bucket < NUM_SAMPLES + 1; bucket++)
    {
        uint_t currentOffset = h_globalBucketOffsets[bucket];
        uint_t bucketLen = currentOffset - previousOffset;

        bitonicSort(dataBuffer + previousOffset, bucketLen, sortOrder);
        previousOffset = currentOffset;
    }
}

/*
Sorts input data with parallel sample sort.
*/
double sortParallel(
    data_t *h_output, data_t *d_dataTable, data_t *d_dataBuffer, data_t *d_samples, uint_t *d_localBucketSizes,
    uint_t *d_localBucketOffsets, uint_t *h_globalBucketOffsets, uint_t *d_globalBucketOffsets, uint_t tableLen,
    order_t sortOrder
)
{
    LARGE_INTEGER timer;
    cudaError_t error;

    startStopwatch(&timer);
    sampleSort(
        d_dataTable, d_dataBuffer, d_samples, h_globalBucketOffsets, d_globalBucketOffsets, d_localBucketSizes,
        d_localBucketOffsets, tableLen, sortOrder
    );

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    double time = endStopwatch(timer);

    error = cudaMemcpy(h_output, d_dataBuffer, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    return time;
}
