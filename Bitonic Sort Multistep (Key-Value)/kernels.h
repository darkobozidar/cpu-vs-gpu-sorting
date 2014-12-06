#ifndef KERNELS_H
#define KERNELS_H


// Bitonic sort
template <order_t sortOrder>
__global__ void bitonicSortKernel(data_t *keys, data_t *values, uint_t tableLen);

// Bitonic multistep merge
template <order_t sortOrder>
__global__ void multiStep1Kernel(data_t *key, data_t *values, int_t tableLen, uint_t step);
template <order_t sortOrder>
__global__ void multiStep2Kernel(data_t *key, data_t *values, int_t tableLen, uint_t step);
template <order_t sortOrder>
__global__ void multiStep3Kernel(data_t *key, data_t *values, int_t tableLen, uint_t step);
template <order_t sortOrder>
__global__ void multiStep4Kernel(data_t *key, data_t *values, int_t tableLen, uint_t step);
//template <order_t sortOrder>
//__global__ void multiStep5Kernel(data_t *table, uint_t tableLen, uint_t step);
//template <order_t sortOrder>
//__global__ void multiStep6Kernel(data_t *table, uint_t tableLen, uint_t step);

// Bitonic merge
template <order_t sortOrder>
__global__ void bitonicMergeGlobalKernel(data_t *keys, data_t *values, uint_t tableLen, uint_t step);
template <order_t sortOrder, bool isFirstStepOfPhase>
__global__ void bitonicMergeLocalKernel(data_t *keys, data_t *values, uint_t tableLen, uint_t step);

#endif
