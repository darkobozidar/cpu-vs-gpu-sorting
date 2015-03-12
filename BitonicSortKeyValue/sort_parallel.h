#ifndef BITONIC_SORT_PARALLEL_KEY_VALUE_H
#define BITONIC_SORT_PARALLEL_KEY_VALUE_H

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"


class BitonicSortParallelKeyValue : public SortParallelKeyValue
{
private:
    std::string _sortName = "Bitonic sort parallel key value";

    template <order_t sortOrder>
    void runBitoicSortKernel(data_t *d_keys, data_t *d_values, uint_t arrayLength);
    template <order_t sortOrder>
    void runBitonicMergeGlobalKernel(data_t *d_keys, data_t *d_values, uint_t arrayLength, uint_t phase, uint_t step);
    template <order_t sortOrder>
    void runBitoicMergeLocalKernel(data_t *d_keys, data_t *values, uint_t arrayLength, uint_t phase, uint_t step);
    template <order_t sortOrder>
    void bitonicSortParallelKeyValue(data_t *d_keys, data_t *d_values, uint_t arrayLength);
    void sortPrivate();

public:
    std::string getSortName()
    {
        return this->_sortName;
    }
};

#endif
