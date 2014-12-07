#ifndef KERNELS_H
#define KERNELS_H

#include "cuda_runtime.h"

#include "../Utils/data_types_common.h"


template <order_t sortOrder>
__global__ void bitonicSortKernel(data_t *dataTable, uint_t tableLen);

//__global__ void initIntervalsKernel(el_t *table, interval_t *intervals, uint_t tableLen, uint_t step,
//                                    uint_t phasesBitonicMerge);
//__global__ void generateIntervalsKernel(el_t *table, interval_t *input, interval_t *output, uint_t tableLen,
//                                        uint_t phase, uint_t step, uint_t phasesBitonicMerge);
//__global__ void bitonicMergeKernel(el_t *input, el_t *output, interval_t *intervals, uint_t phase, bool orderAsc);

#endif
