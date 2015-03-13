#ifndef BITONIC_SORT_PARALLEL_H
#define BITONIC_SORT_PARALLEL_H

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"
#include "../BitonicSortMultistepParallel/constants.h"


/*
Due to (extreme =)) optimization code for key only and key-value sorts are entirely separated.
TODO once the testing is done merge common code for key only and key-value sorts.
*/
class BitonicSortParallel : public SortParallel
{
protected:
    std::string _sortName = "Bitonic sort parallel";

	// Key only
    template <order_t sortOrder, uint_t threadsBitonicSort, uint_t elemsThreadBitonicSort>
    void runBitoicSortKernel(data_t *keys, uint_t tableLen);
    template <order_t sortOrder, uint_t threadsMerge, uint_t elemsThreadMerge>
    void runBitonicMergeGlobalKernel(data_t *d_keys, uint_t arrayLength, uint_t phase, uint_t step);
    template <order_t sortOrder, uint_t threadsMerge, uint_t elemsThreadMerge>
    void runBitoicMergeLocalKernel(data_t *d_keys, uint_t arrayLength, uint_t phase, uint_t step);
    template <order_t sortOrder>
    void bitonicSortParallel(data_t *d_keys, uint_t arrayLength);
    void sortKeyOnly();

    // Key-value
    template <order_t sortOrder, uint_t threadsBitonicSort, uint_t elemsThreadBitonicSort>
    void runBitoicSortKernel(data_t *d_keys, data_t *d_values, uint_t arrayLength);
    template <order_t sortOrder, uint_t threadsMerge, uint_t elemsThreadMerge>
    void runBitonicMergeGlobalKernel(data_t *d_keys, data_t *d_values, uint_t arrayLength, uint_t phase, uint_t step);
    template <order_t sortOrder, uint_t threadsMerge, uint_t elemsThreadMerge>
    void runBitoicMergeLocalKernel(data_t *d_keys, data_t *values, uint_t arrayLength, uint_t phase, uint_t step);
    template <order_t sortOrder>
    void bitonicSortParallel(data_t *d_keys, data_t *d_values, uint_t arrayLength);
    void sortKeyValue();

public:
    std::string getSortName()
    {
        return this->_sortName;
    }
};


/* BITONIC SORT MULTISTEP PARALLEL KEY ONLY */

template void BitonicSortParallel::runBitoicSortKernel
    <ORDER_ASC, THREADS_PER_BITONIC_SORT_KO, ELEMS_PER_THREAD_BITONIC_SORT_KO>(
    data_t *d_keys, uint_t arrayLength
);
template void BitonicSortParallel::runBitoicSortKernel
    <ORDER_DESC, THREADS_PER_BITONIC_SORT_KO, ELEMS_PER_THREAD_BITONIC_SORT_KO>(
    data_t *d_keys, uint_t arrayLength
);

template void BitonicSortParallel::runBitonicMergeGlobalKernel
    <ORDER_ASC, THREADS_PER_GLOBAL_MERGE_KO, ELEMS_PER_THREAD_GLOBAL_MERGE_KO>(
    data_t *d_keys, uint_t arrayLength, uint_t phase, uint_t step
);
template void BitonicSortParallel::runBitonicMergeGlobalKernel
    <ORDER_DESC, THREADS_PER_GLOBAL_MERGE_KO, ELEMS_PER_THREAD_GLOBAL_MERGE_KO>(
    data_t *d_keys, uint_t arrayLength, uint_t phase, uint_t step
);

template void BitonicSortParallel::runBitoicMergeLocalKernel
    <ORDER_ASC, THREADS_PER_LOCAL_MERGE_KO, ELEMS_PER_THREAD_LOCAL_MERGE_KO>(
    data_t *d_keys, uint_t arrayLength, uint_t phase, uint_t step
);
template void BitonicSortParallel::runBitoicMergeLocalKernel
    <ORDER_DESC, THREADS_PER_LOCAL_MERGE_KO, ELEMS_PER_THREAD_LOCAL_MERGE_KO>(
    data_t *d_keys, uint_t arrayLength, uint_t phase, uint_t step
);


/* BITONIC SORT MULTISTEP PARALLEL KEY VALUE */

template void BitonicSortParallel::runBitoicSortKernel
    <ORDER_ASC, THREADS_PER_BITONIC_SORT_KV, ELEMS_PER_THREAD_BITONIC_SORT_KV>(
    data_t *d_keys, data_t *values, uint_t arrayLength
);
template void BitonicSortParallel::runBitoicSortKernel
    <ORDER_DESC, THREADS_PER_BITONIC_SORT_KV, ELEMS_PER_THREAD_BITONIC_SORT_KV>(
    data_t *d_keys, data_t *values, uint_t arrayLength
);

template void BitonicSortParallel::runBitonicMergeGlobalKernel
    <ORDER_ASC, THREADS_PER_GLOBAL_MERGE_KV, ELEMS_PER_THREAD_GLOBAL_MERGE_KV>(
    data_t *d_keys, data_t *values, uint_t arrayLength, uint_t phase, uint_t step
);
template void BitonicSortParallel::runBitonicMergeGlobalKernel
    <ORDER_DESC, THREADS_PER_GLOBAL_MERGE_KV, ELEMS_PER_THREAD_GLOBAL_MERGE_KV>(
    data_t *d_keys, data_t *values, uint_t arrayLength, uint_t phase, uint_t step
);

template void BitonicSortParallel::runBitoicMergeLocalKernel
    <ORDER_ASC, THREADS_PER_LOCAL_MERGE_KV, ELEMS_PER_THREAD_LOCAL_MERGE_KV>(
    data_t *d_keys, data_t *values, uint_t arrayLength, uint_t phase, uint_t step
);
template void BitonicSortParallel::runBitoicMergeLocalKernel
    <ORDER_DESC, THREADS_PER_LOCAL_MERGE_KV, ELEMS_PER_THREAD_LOCAL_MERGE_KV>(
    data_t *d_keys, data_t *values, uint_t arrayLength, uint_t phase, uint_t step
);

#endif
