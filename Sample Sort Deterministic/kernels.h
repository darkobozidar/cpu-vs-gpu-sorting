#ifndef KERNELS_H
#define KERNELS_H

__global__ void printElemsKernel(el_t *table, uint_t tableLen);
__global__ void printDataKernel(uint_t *table, uint_t tableLen);

template <typename T>
__global__ void bitonicSortCollectSamplesKernel(el_t *dataTable, data_t *localSamples, uint_t tableLen,
                                                order_t sortOrder);
//__global__ void bitonicMergeGlobalKernel(el_t *dataTable, uint_t tableLen, uint_t step, bool firstStepOfPhase,
//                                         order_t sortOrder);
//__global__ void bitonicMergeLocalKernel(el_t *table, uint_t tableLen, uint_t step, bool isFirstStepOfPhase,
//                                        order_t sortOrder);

#endif
