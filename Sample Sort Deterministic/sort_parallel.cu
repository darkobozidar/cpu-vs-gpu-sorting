#include <stdio.h>
#include <climits>
#include <math.h>
#include <Windows.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cudpp.h>

#include "data_types.h"
#include "constants.h"
#include "utils_cuda.h"
#include "utils_host.h"
#include "kernels.h"


/*
Initializes DEVICE memory needed for paralel sort implementation.
*/
void memoryInit(el_t *h_input, el_t **d_dataInput, el_t **d_dataBuffer, data_t **d_samples,
                uint_t **d_localBucketSizes, uint_t **d_localBucketOffsets, uint_t tableLen,
                uint_t localSamplesLen, uint_t localBucketsLen) {
    cudaError_t error;

    // Data memory allocation
    error = cudaMalloc(d_dataInput, tableLen * sizeof(**d_dataInput));
    checkCudaError(error);
    error = cudaMalloc(d_dataBuffer, tableLen * sizeof(**d_dataBuffer));
    checkCudaError(error);
    error = cudaMalloc(d_samples, localSamplesLen * sizeof(**d_samples));
    checkCudaError(error);
    error = cudaMalloc(d_localBucketSizes, localBucketsLen * sizeof(**d_localBucketSizes));
    checkCudaError(error);
    error = cudaMalloc(d_localBucketOffsets, localBucketsLen * sizeof(**d_localBucketOffsets));
    checkCudaError(error);

    error = cudaMemcpy(*d_dataInput, h_input, tableLen * sizeof(**d_dataInput), cudaMemcpyHostToDevice);
    checkCudaError(error);
}

void cudppInitScan(CUDPPHandle *scanPlan, uint_t tableLen) {
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
Sorts sub-blocks of input data with bitonic sort.
*/
void runBitonicSortCollectSamplesKernel(el_t *dataTable, data_t *samples, uint_t tableLen, order_t sortOrder) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_BITONIC_SORT, 1, 1);

    startStopwatch(&timer);
    bitonicSortCollectSamplesKernel<el_t><<<dimGrid, dimBlock, elemsPerThreadBlock * sizeof(*dataTable)>>>(
        dataTable, samples, tableLen, sortOrder
    );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing bitonic sort kernel");*/
}

template <typename T>
void runBitonicMergeGlobalKernel(T *dataTable, uint_t tableLen, uint_t phase, uint_t step, order_t sortOrder) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t elemsPerThreadBlock = THREADS_PER_GLOBAL_MERGE * ELEMS_PER_THREAD_GLOBAL_MERGE;
    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_GLOBAL_MERGE, 1, 1);

    startStopwatch(&timer);
    bitonicMergeGlobalKernel<T><<<dimGrid, dimBlock>>>(dataTable, tableLen, step, step == phase, sortOrder);
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing bitonic merge global kernel");*/
}

template <typename T>
void runBitoicMergeLocalKernel(T *dataTable, uint_t tableLen, uint_t phase, uint_t step, order_t sortOrder) {
    cudaError_t error;
    LARGE_INTEGER timer;

    // Every thread loads and sorts 2 elements
    uint_t elemsPerThreadBlock = THREADS_PER_LOCAL_MERGE * ELEMS_PER_THREAD_LOCAL_MERGE;
    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_LOCAL_MERGE, 1, 1);

    startStopwatch(&timer);
    bitonicMergeLocalKernel<<<dimGrid, dimBlock>>>(
        dataTable, tableLen, step, phase == step, sortOrder
    );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing bitonic merge local kernel");*/
}

void runCollectGlobalSamplesKernel(data_t *samples, uint_t samplesLen) {
    LARGE_INTEGER timer;

    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(NUM_SAMPLES, 1, 1);

    startStopwatch(&timer);
    collectGlobalSamplesKernel<<<dimGrid, dimBlock>>>(samples, samplesLen);
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing kernel for collection of global samples");*/
}

void runSampleIndexingKernel(el_t *dataTable, data_t *samples, data_t *bucketSizes, uint_t tableLen,
                             uint_t numAllBuckets, order_t sortOrder) {
    LARGE_INTEGER timer;

    // TODO comment
    dim3 dimGrid((numAllBuckets - 1) / (THREADS_PER_SAMPLE_INDEXING / NUM_SAMPLES * (NUM_SAMPLES + 1)) + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_SAMPLE_INDEXING, 1, 1);

    startStopwatch(&timer);
    sampleIndexingKernel<<<dimGrid, dimBlock>>>(dataTable, samples, bucketSizes, tableLen, sortOrder);
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing kernel sample indexing");*/
}

void runPrintElemsKernel(el_t *table, uint_t tableLen) {
    printElemsKernel<<<1, 1>>>(table, tableLen);
    cudaError_t error = cudaDeviceSynchronize();
    checkCudaError(error);
}

void runPrintDataKernel(data_t *table, uint_t tableLen) {
    printDataKernel<<<1, 1>>>(table, tableLen);
    cudaError_t error = cudaDeviceSynchronize();
    checkCudaError(error);
}

template <typename T>
void bitonicMerge(T *dataTable, uint_t tableLen, uint_t elemsPerBlockBitonicSort, order_t sortOrder) {
    uint_t tableLenPower2 = nextPowerOf2(tableLen);
    uint_t elemsPerBlockMergeLocal = THREADS_PER_LOCAL_MERGE * ELEMS_PER_THREAD_LOCAL_MERGE;

    // Number of phases, which can be executed in shared memory (stride is lower than shared memory size)
    uint_t phasesBitonicSort = log2((double)min(tableLenPower2, elemsPerBlockBitonicSort));
    uint_t phasesMergeLocal = log2((double)min(tableLenPower2, elemsPerBlockMergeLocal));
    uint_t phasesAll = log2((double)tableLenPower2);

    for (uint_t phase = phasesBitonicSort + 1; phase <= phasesAll; phase++) {
        uint_t step = phase;
        while (step > phasesMergeLocal) {
            runBitonicMergeGlobalKernel<T>(dataTable, tableLen, phase, step, sortOrder);
            step--;
        }

        runBitoicMergeLocalKernel<T>(dataTable, tableLen, phase, step, sortOrder);
    }
}

el_t* sampleSort(el_t *dataTable, el_t *dataBuffer, data_t *samples, uint_t *d_localBucketSizes,
                 uint_t *d_localBucketOffsets, uint_t tableLen, uint_t localSamplesLen, uint_t localBucketsLen,
                 order_t sortOrder) {
    CUDPPHandle scanPlan;

    // TODO Should this be done before or after stopwatch?
    cudppInitScan(&scanPlan, localBucketsLen);
    runBitonicSortCollectSamplesKernel(dataTable, samples, tableLen, sortOrder);

    uint_t elemsPerBlockBitonicSort = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    if (tableLen <= elemsPerBlockBitonicSort) {
        return dataTable;
    }

    bitonicMerge<data_t>(samples, localSamplesLen, NUM_SAMPLES, sortOrder);
    // TODO handle case, if all samples are the same
    runCollectGlobalSamplesKernel(samples, localSamplesLen);
    runPrintDataKernel(samples, NUM_SAMPLES);
    runSampleIndexingKernel(dataTable, samples, d_localBucketSizes, tableLen, localBucketsLen, sortOrder);

    CUDPPResult result = cudppScan(scanPlan, d_localBucketOffsets, d_localBucketSizes, localBucketsLen);
    if (result != CUDPP_SUCCESS) {
        printf("Error in cudppScan()\n");
        getchar();
        exit(-1);
    }

    runPrintDataKernel(d_localBucketSizes, localBucketsLen);
    runPrintDataKernel(d_localBucketOffsets, localBucketsLen);

    // TODO other steps
    return dataTable;
}

void sortParallel(el_t *h_dataInput, el_t *h_dataOutput, uint_t tableLen, order_t sortOrder) {
    el_t *d_dataInput, *d_dataBuffer, *d_dataResult;
    // First it holds LOCAL and than GLOBAL samples
    data_t *d_samples;
    uint_t *d_globalBucketSizes, *d_localBucketSizes, *d_localBucketOffsets;

    uint_t elemsPerInitBitonicSort = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    uint_t localSamplesDistance = (THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT) / NUM_SAMPLES;
    uint_t localSamplesLen = (tableLen - 1) / localSamplesDistance + 1;
    // (number of all data blocks (tiles)) * (number buckets generated from NUM_SAMPLES)
    uint_t localBucketsLen = ((tableLen - 1) / elemsPerInitBitonicSort + 1) * (NUM_SAMPLES + 1);

    LARGE_INTEGER timer;
    cudaError_t error;

    memoryInit(
        h_dataInput, &d_dataInput, &d_dataBuffer, &d_samples, &d_localBucketSizes, &d_localBucketOffsets,
        tableLen, localSamplesLen, localBucketsLen
    );

    startStopwatch(&timer);
    d_dataResult = sampleSort(
        d_dataInput, d_dataBuffer, d_samples, d_localBucketSizes, d_localBucketOffsets, tableLen,
        localSamplesLen, localBucketsLen, sortOrder
    );

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    double time = endStopwatch(timer, "Executing parallel sample sort.");
    printf("Operations (pair swaps): %.2f M/s\n", tableLen / 500.0 / time);

    error = cudaMemcpy(h_dataOutput, d_dataResult, tableLen * sizeof(*h_dataOutput), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    cudaFree(d_dataInput);
    cudaFree(d_dataBuffer);
    cudaFree(d_samples);
}
