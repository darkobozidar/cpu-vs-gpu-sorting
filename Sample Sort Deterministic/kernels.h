#ifndef KERNELS_H
#define KERNELS_H

__global__ void printElemsKernel(el_t *table, uint_t tableLen);
__global__ void printDataKernel(uint_t *table, uint_t tableLen);

template <typename T>
__global__ void bitonicSortCollectSamplesKernel(
    T *dataTable, data_t *localSamples, uint_t tableLen, order_t sortOrder
);
template <typename T>
__global__ void bitonicMergeGlobalKernel(
    T *dataTable, uint_t tableLen, uint_t step, bool firstStepOfPhase, order_t sortOrder
);
template <typename T>
__global__ void bitonicMergeLocalKernel(
    T *table, uint_t tableLen, uint_t step, bool isFirstStepOfPhase, order_t sortOrder
);
__global__ void collectGlobalSamplesKernel(data_t *samples, uint_t samplesLen);
__global__ void sampleIndexingKernel(
    el_t *dataTable, const data_t* __restrict__ samples, uint_t *bucketSizes, uint_t tableLen,
    order_t sortOrder
);
__global__ void bucketsRelocationKernel(
    el_t *dataTable, el_t *dataBuffer, uint_t *d_globalBucketOffsets, const uint_t* __restrict__ localBucketSizes,
    const uint_t* __restrict__ localBucketOffsets, uint_t tableLen
);

#endif
