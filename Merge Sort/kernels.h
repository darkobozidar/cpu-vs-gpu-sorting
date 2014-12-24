#ifndef KERNELS_H
#define KERNELS_H


template <order_t sortOrder>
__global__ void mergeSortKernel(data_t *input);

template <order_t sortOrder>
__global__ void generateSamplesKernel(data_t *dataTable, sample_t *samples, uint_t sortedBlockSize);

//__global__ void generateRanksKernel(el_t* table, el_t *samples, uint_t *ranksEven, uint_t *ranksOdd,
//                                    uint_t sortedBlockSize, bool orderAsc);
//__global__ void mergeKernel(el_t* input, el_t* output, uint_t *ranksEven, uint_t *ranksOdd, uint_t tableLen,
//                            uint_t sortedBlockSize);

#endif
