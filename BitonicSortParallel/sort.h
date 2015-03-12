#ifndef BITONIC_SORT_PARALLEL_H
#define BITONIC_SORT_PARALLEL_H

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"


/*
Due to (extreme =)) optimization code for key only and key-value sorts are entirely separated.
TODO once the testing is done merge common code for key only and key-value sorts.
*/
class BitonicSortParallel : public SortParallel
{
private:
    std::string _sortName = "Bitonic sort parallel";

	// Key only
    template <order_t sortOrder>
    void runBitoicSortKernel(data_t *keys, uint_t tableLen);
    template <order_t sortOrder>
    void runBitonicMergeGlobalKernel(data_t *d_keys, uint_t arrayLength, uint_t phase, uint_t step);
    template <order_t sortOrder>
    void runBitoicMergeLocalKernel(data_t *d_keys, uint_t arrayLength, uint_t phase, uint_t step);
    template <order_t sortOrder>
    void bitonicSortParallel(data_t *d_keys, uint_t arrayLength);
    void sortKeyOnly();

    // Key-value
    template <order_t sortOrder>
    void runBitoicSortKernel(data_t *d_keys, data_t *d_values, uint_t arrayLength);
    template <order_t sortOrder>
    void runBitonicMergeGlobalKernel(data_t *d_keys, data_t *d_values, uint_t arrayLength, uint_t phase, uint_t step);
    template <order_t sortOrder>
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

#endif
