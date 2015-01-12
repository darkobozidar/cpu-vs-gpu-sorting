#ifndef KERNELS_H
#define KERNELS_H


// Bitonic sort kernels
template <order_t sortOrder>
__global__ void bitonicSortCollectSamplesKernel(data_t *dataTable, data_t *localSamples, uint_t tableLen);
//template <typename T>
//__global__ void bitonicSortKernel(T *dataTable, uint_t tableLen, order_t sortOrder);

// Bitonic merge kernels
template <order_t sortOrder>
__global__ void bitonicMergeGlobalKernel(
    data_t *dataTable, uint_t tableLen, uint_t step, bool firstStepOfPhase
);
template <order_t sortOrder>
__global__ void bitonicMergeLocalKernel(
    data_t *dataTable, uint_t tableLen, uint_t step, bool isFirstStepOfPhase
);

//__global__ void collectGlobalSamplesKernel(data_t *samples, uint_t samplesLen);
//__global__ void sampleIndexingKernel(
//    el_t *dataTable, const data_t* __restrict__ samples, uint_t *bucketSizes, uint_t tableLen,
//    order_t sortOrder
//);
//__global__ void bucketsRelocationKernel(
//    el_t *dataTable, el_t *dataBuffer, uint_t *d_globalBucketOffsets, const uint_t* __restrict__ localBucketSizes,
//    const uint_t* __restrict__ localBucketOffsets, uint_t tableLen
//);

#endif
