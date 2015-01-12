#include <stdio.h>
#include <math.h>
#include <Windows.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"
#include "../Utils/cuda.h"
#include "../Utils/host.h"
#include "constants.h"
#include "kernels.h"


///*
//Initializes DEVICE memory needed for paralel sort implementation.
//*/
//void memoryInit(el_t *h_input, el_t **d_dataInput, el_t **d_dataBuffer, data_t **d_samples,
//                uint_t **d_globalBucketOffsets, uint_t **d_localBucketSizes, uint_t **d_localBucketOffsets,
//                uint_t tableLen, uint_t localSamplesLen, uint_t localBucketsLen) {
//    cudaError_t error;
//
//    // Data memory allocation
//    error = cudaMalloc(d_dataInput, tableLen * sizeof(**d_dataInput));
//    checkCudaError(error);
//    error = cudaMalloc(d_dataBuffer, tableLen * sizeof(**d_dataBuffer));
//    checkCudaError(error);
//    error = cudaMalloc(d_samples, localSamplesLen * sizeof(**d_samples));
//    checkCudaError(error);
//    error = cudaMalloc(d_globalBucketOffsets, (NUM_SAMPLES + 1) * sizeof(**d_globalBucketOffsets));
//    checkCudaError(error);
//    error = cudaMalloc(d_localBucketSizes, localBucketsLen * sizeof(**d_localBucketSizes));
//    checkCudaError(error);
//    error = cudaMalloc(d_localBucketOffsets, localBucketsLen * sizeof(**d_localBucketOffsets));
//    checkCudaError(error);
//
//    error = cudaMemcpy(*d_dataInput, h_input, tableLen * sizeof(**d_dataInput), cudaMemcpyHostToDevice);
//    checkCudaError(error);
//}
//
///*
//Initializes CUDPP scan.
//*/
//void cudppInitScan(CUDPPHandle *scanPlan, uint_t tableLen) {
//    // Initializes the CUDPP Library
//    CUDPPHandle theCudpp;
//    cudppCreate(&theCudpp);
//
//    CUDPPConfiguration config;
//    config.op = CUDPP_ADD;
//    config.datatype = CUDPP_UINT;
//    config.algorithm = CUDPP_SCAN;
//    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
//
//    *scanPlan = 0;
//    CUDPPResult result = cudppPlan(theCudpp, scanPlan, config, tableLen, 1, 0);
//
//    if (result != CUDPP_SUCCESS) {
//        printf("Error creating CUDPPPlan\n");
//        getchar();
//        exit(-1);
//    }
//}
//
///*
//Sorts sub-blocks of input data with bitonic sort.
//*/
//void runBitonicSortCollectSamplesKernel(el_t *dataTable, data_t *samples, uint_t tableLen, order_t sortOrder) {
//    cudaError_t error;
//    LARGE_INTEGER timer;
//
//    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
//    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
//    dim3 dimBlock(THREADS_PER_BITONIC_SORT, 1, 1);
//
//    startStopwatch(&timer);
//    bitonicSortCollectSamplesKernel<el_t><<<dimGrid, dimBlock, elemsPerThreadBlock * sizeof(*dataTable)>>>(
//        dataTable, samples, tableLen, sortOrder
//    );
//    /*error = cudaDeviceSynchronize();
//    checkCudaError(error);
//    endStopwatch(timer, "Executing kernel for bitonic sort and collecting samples");*/
//}
//
///*
//Sorts sub-blocks of input data with bitonic sort.
//*/
//void runBitonicSortKernel(el_t *dataTable, uint_t tableLen, order_t sortOrder) {
//    cudaError_t error;
//    LARGE_INTEGER timer;
//
//    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
//    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
//    dim3 dimBlock(THREADS_PER_BITONIC_SORT, 1, 1);
//
//    startStopwatch(&timer);
//    bitonicSortKernel<el_t><<<dimGrid, dimBlock, elemsPerThreadBlock * sizeof(*dataTable)>>>(
//        dataTable, tableLen, sortOrder
//    );
//    /*error = cudaDeviceSynchronize();
//    checkCudaError(error);
//    endStopwatch(timer, "Executing bitonic sort kernel");*/
//}
//
//template <typename T>
//void runBitonicMergeGlobalKernel(T *dataTable, uint_t tableLen, uint_t phase, uint_t step, order_t sortOrder) {
//    cudaError_t error;
//    LARGE_INTEGER timer;
//
//    uint_t elemsPerThreadBlock = THREADS_PER_GLOBAL_MERGE * ELEMS_PER_THREAD_GLOBAL_MERGE;
//    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
//    dim3 dimBlock(THREADS_PER_GLOBAL_MERGE, 1, 1);
//
//    startStopwatch(&timer);
//    bitonicMergeGlobalKernel<T><<<dimGrid, dimBlock>>>(dataTable, tableLen, step, step == phase, sortOrder);
//    /*error = cudaDeviceSynchronize();
//    checkCudaError(error);
//    endStopwatch(timer, "Executing bitonic merge global kernel");*/
//}
//
//template <typename T>
//void runBitoicMergeLocalKernel(T *dataTable, uint_t tableLen, uint_t phase, uint_t step, order_t sortOrder) {
//    cudaError_t error;
//    LARGE_INTEGER timer;
//
//    // Every thread loads and sorts 2 elements
//    uint_t elemsPerThreadBlock = THREADS_PER_LOCAL_MERGE * ELEMS_PER_THREAD_LOCAL_MERGE;
//    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
//    dim3 dimBlock(THREADS_PER_LOCAL_MERGE, 1, 1);
//
//    startStopwatch(&timer);
//    bitonicMergeLocalKernel<<<dimGrid, dimBlock>>>(
//        dataTable, tableLen, step, phase == step, sortOrder
//    );
//    /*error = cudaDeviceSynchronize();
//    checkCudaError(error);
//    endStopwatch(timer, "Executing bitonic merge local kernel");*/
//}
//
///*
//From all LOCAL samples collects (NUM_SAMPLES) GLOBAL samples.
//*/
//void runCollectGlobalSamplesKernel(data_t *samples, uint_t samplesLen) {
//    LARGE_INTEGER timer;
//
//    dim3 dimGrid(1, 1, 1);
//    dim3 dimBlock(NUM_SAMPLES, 1, 1);
//
//    startStopwatch(&timer);
//    collectGlobalSamplesKernel<<<dimGrid, dimBlock>>>(samples, samplesLen);
//    /*error = cudaDeviceSynchronize();
//    checkCudaError(error);
//    endStopwatch(timer, "Executing kernel for collection of global samples");*/
//}
//
///*
//For every sample searches, how many elements in tile are lower than it's value.
//*/
//void runSampleIndexingKernel(el_t *dataTable, data_t *samples, data_t *bucketSizes, uint_t tableLen,
//                             uint_t numAllBuckets, order_t sortOrder) {
//    LARGE_INTEGER timer;
//
//    // Number of threads per thread block can be greater than number of samples.
//    uint_t elemsPerBitonicSort = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
//    uint_t numBlocks = (tableLen - 1) / elemsPerBitonicSort + 1;
//    uint_t threadBlockSize = min(numBlocks * NUM_SAMPLES, THREADS_PER_SAMPLE_INDEXING);
//
//    // Every thread block creates from NUM_SAMPLES samples (NUM_SAMPLES + 1) buckets
//    dim3 dimGrid((numAllBuckets - 1) / (threadBlockSize / NUM_SAMPLES * (NUM_SAMPLES + 1)) + 1, 1, 1);
//    dim3 dimBlock(threadBlockSize, 1, 1);
//
//    startStopwatch(&timer);
//    sampleIndexingKernel<<<dimGrid, dimBlock>>>(dataTable, samples, bucketSizes, tableLen, sortOrder);
//    /*error = cudaDeviceSynchronize();
//    checkCudaError(error);
//    endStopwatch(timer, "Executing kernel sample indexing");*/
//}
//
///*
//From local bucket sizes and offsets scatters elements to their global buckets. At the end it coppies
//global bucket sizes (sizes of whole buckets, not just bucket size per tile (local size)) to host.
//*/
//void runBucketsRelocationKernel(el_t *dataTable, el_t *dataBuffer, uint_t *h_globalBucketOffsets,
//                                uint_t *d_globalBucketOffsets, uint_t *localBucketSizes,
//                                uint_t *localBucketOffsets, uint_t tableLen) {
//    // For NUM_SAMPLES samples (NUM_SAMPLES + 1) buckets are created
//    uint_t sharedMemSize = 2 * (NUM_SAMPLES + 1);
//    uint_t elemsPerBitonicSort = THREADS_PER_GLOBAL_MERGE * ELEMS_PER_THREAD_GLOBAL_MERGE;
//    LARGE_INTEGER timer;
//    cudaError_t error;
//
//    dim3 dimGrid((tableLen - 1) / elemsPerBitonicSort + 1, 1, 1);
//    dim3 dimBlock(THREADS_PER_BUCKETS_RELOCATION, 1, 1);
//    bucketsRelocationKernel<<<dimGrid, dimBlock, sharedMemSize * sizeof(*localBucketSizes)>>>(
//        dataTable, dataBuffer, d_globalBucketOffsets, localBucketSizes, localBucketOffsets, tableLen
//    );
//
//    /*error = cudaDeviceSynchronize();
//    checkCudaError(error);
//    endStopwatch(timer, "Executing kernel for buckets relocation");*/
//
//    error = cudaMemcpy(
//        h_globalBucketOffsets, d_globalBucketOffsets, (NUM_SAMPLES + 1) * sizeof(*h_globalBucketOffsets),
//        cudaMemcpyDeviceToHost
//    );
//    checkCudaError(error);
//}
//
//void runPrintElemsKernel(el_t *table, uint_t tableLen) {
//    printElemsKernel<<<1, 1>>>(table, tableLen);
//    cudaError_t error = cudaDeviceSynchronize();
//    checkCudaError(error);
//}
//
//void runPrintDataKernel(data_t *table, uint_t tableLen) {
//    printDataKernel<<<1, 1>>>(table, tableLen);
//    cudaError_t error = cudaDeviceSynchronize();
//    checkCudaError(error);
//}
//
///*
//Performs global bitonic merge, when number of elements is greater than shared memory size.
//*/
//template <typename T>
//void bitonicMerge(T *dataTable, uint_t tableLen, uint_t elemsPerBlockBitonicSort, order_t sortOrder) {
//    uint_t tableLenPower2 = nextPowerOf2(tableLen);
//    uint_t elemsPerBlockMergeLocal = THREADS_PER_LOCAL_MERGE * ELEMS_PER_THREAD_LOCAL_MERGE;
//
//    // Number of phases, which can be executed in shared memory (stride is lower than shared memory size)
//    uint_t phasesBitonicSort = log2((double)min(tableLenPower2, elemsPerBlockBitonicSort));
//    uint_t phasesMergeLocal = log2((double)min(tableLenPower2, elemsPerBlockMergeLocal));
//    uint_t phasesAll = log2((double)tableLenPower2);
//
//    for (uint_t phase = phasesBitonicSort + 1; phase <= phasesAll; phase++) {
//        uint_t step = phase;
//        while (step > phasesMergeLocal) {
//            runBitonicMergeGlobalKernel<T>(dataTable, tableLen, phase, step, sortOrder);
//            step--;
//        }
//
//        runBitoicMergeLocalKernel<T>(dataTable, tableLen, phase, step, sortOrder);
//    }
//}
//
///*
//Performs bitonic sort.
//*/
//template <typename T>
//void bitonicSort(T *dataTable, uint_t tableLen, order_t sortOrder) {
//    uint_t tableLenPower2 = nextPowerOf2(tableLen);
//    uint_t elemsPerBlockBitonicSort = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
//
//    uint_t phasesAll = log2((double)tableLenPower2);
//
//    runBitonicSortKernel(dataTable, tableLen, sortOrder);
//
//    bitonicMerge<T>(dataTable, tableLen, elemsPerBlockBitonicSort, sortOrder);
//}
//
//// TODO figure out what the bottleneck is
//el_t* sampleSort(el_t *dataTable, el_t *dataBuffer, data_t *samples, uint_t *h_globalBucketOffsets,
//                 uint_t *d_globalBucketOffsets, uint_t *d_localBucketSizes, uint_t *d_localBucketOffsets,
//                 uint_t tableLen, uint_t localSamplesLen, uint_t localBucketsLen, order_t sortOrder) {
//    CUDPPHandle scanPlan;
//
//    // TODO Should this be done before or after stopwatch?
//    cudppInitScan(&scanPlan, localBucketsLen);
//    runBitonicSortCollectSamplesKernel(dataTable, samples, tableLen, sortOrder);
//
//    uint_t elemsPerBlockBitonicSort = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
//    if (tableLen <= elemsPerBlockBitonicSort) {
//        return dataTable;
//    }
//
//    // Local samples are already partially ordered - NUM_SAMPLES per every tile. These partially ordered
//    // samples need to be merged.
//    bitonicMerge<data_t>(samples, localSamplesLen, NUM_SAMPLES, sortOrder);
//    // TODO handle case, if all samples are the same
//    runCollectGlobalSamplesKernel(samples, localSamplesLen);
//    runSampleIndexingKernel(dataTable, samples, d_localBucketSizes, tableLen, localBucketsLen, sortOrder);
//
//    CUDPPResult result = cudppScan(scanPlan, d_localBucketOffsets, d_localBucketSizes, localBucketsLen);
//    if (result != CUDPP_SUCCESS) {
//        printf("Error in cudppScan()\n");
//        getchar();
//        exit(-1);
//    }
//
//    runBucketsRelocationKernel(
//        dataTable, dataBuffer, h_globalBucketOffsets, d_globalBucketOffsets, d_localBucketSizes,
//        d_localBucketOffsets, tableLen
//    );
//
//    // Sorts every bucket with bitonic sort
//    uint_t previousOffset = 0;
//    for (uint_t bucket = 0; bucket < NUM_SAMPLES + 1; bucket++) {
//        uint_t currentOffset = h_globalBucketOffsets[bucket];
//        uint_t bucketLen = currentOffset - previousOffset;
//
//        bitonicSort(dataBuffer, bucketLen, sortOrder);
//    }
//
//    return dataBuffer;
//}

double sortParallel(
    data_t *h_output, data_t *d_dataTable, data_t *d_dataBuffer, data_t *d_samples, uint_t *d_localBucketSizes,
    uint_t *d_localBucketOffsets, uint_t *h_globalBucketOffsets, uint_t *d_globalBucketOffsets, uint_t tableLen,
    order_t sortOrder
)
{
    //uint_t elemsPerInitBitonicSort = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    //uint_t localSamplesDistance = (THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT) / NUM_SAMPLES;
    //uint_t localSamplesLen = (tableLen - 1) / localSamplesDistance + 1;
    //// (number of all data blocks (tiles)) * (number buckets generated from NUM_SAMPLES)
    //uint_t localBucketsLen = ((tableLen - 1) / elemsPerInitBitonicSort + 1) * (NUM_SAMPLES + 1);

    LARGE_INTEGER timer;
    cudaError_t error;

    //memoryInit(
    //    h_dataInput, &d_dataInput, &d_dataBuffer, &d_samples, &d_globalBucketOffsets, &d_localBucketSizes,
    //    &d_localBucketOffsets, tableLen, localSamplesLen, localBucketsLen
    //);

    //startStopwatch(&timer);
    //d_dataResult = sampleSort(
    //    d_dataInput, d_dataBuffer, d_samples, h_globalBucketOffsets, d_globalBucketOffsets, d_localBucketSizes,
    //    d_localBucketOffsets, tableLen, localSamplesLen, localBucketsLen, sortOrder
    //);

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    double time = endStopwatch(timer);

    error = cudaMemcpy(h_output, d_dataTable, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    return time;
}
