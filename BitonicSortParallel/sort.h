#ifndef BITONIC_SORT_PARALLEL_H
#define BITONIC_SORT_PARALLEL_H

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"


class BitonicSortParallel : public SortParallel
{
private:
    std::string _sortName = "Bitonic sort parallel";

	// Key only
    template <order_t sortOrder>
    void runBitoicSortKernelKeyOnly(data_t *keys, uint_t tableLen);
    template <order_t sortOrder>
    void runBitonicMergeGlobalKernelKeyOnly(data_t *d_keys, uint_t arrayLength, uint_t phase, uint_t step);
    template <order_t sortOrder>
    void runBitoicMergeLocalKernelKeyOnly(data_t *d_keys, uint_t arrayLength, uint_t phase, uint_t step);
    template <order_t sortOrder>
    void bitonicSortParallelKeyOnly(data_t *d_keys, uint_t arrayLength);
    void sortKeyOnly();

    // Key-value
    template <order_t sortOrder>
    void runBitoicSortKernelKeyValue(data_t *d_keys, data_t *d_values, uint_t arrayLength);
    template <order_t sortOrder>
    void runBitonicMergeGlobalKernelKeyValue(data_t *d_keys, data_t *d_values, uint_t arrayLength, uint_t phase, uint_t step);
    template <order_t sortOrder>
    void runBitoicMergeLocalKernelKeyValue(data_t *d_keys, data_t *values, uint_t arrayLength, uint_t phase, uint_t step);
    template <order_t sortOrder>
    void bitonicSortParallelKeyValue(data_t *d_keys, data_t *d_values, uint_t arrayLength);
    void sortKeyValue();

public:
    std::string getSortName()
    {
        return this->_sortName;
    }
};

#endif
