#ifndef KERNELS_H
#define KERNELS_H


template <order_t sortOrder>
__global__ void bitonicSortKernel(data_t *dataTable, uint_t tableLen);

//__global__ void multiStep1Kernel(el_t *table, uint_t phase, uint_t step, bool orderAsc);
//__global__ void multiStep2Kernel(el_t *table, uint_t phase, uint_t step, bool orderAsc);
//__global__ void multiStep3Kernel(el_t *table, uint_t phase, uint_t step, bool orderAsc);
//__global__ void multiStep4Kernel(el_t *table, uint_t phase, uint_t step, bool orderAsc);
//__global__ void bitonicMergeKernel(el_t *table, uint_t phase, bool orderAsc);

#endif
