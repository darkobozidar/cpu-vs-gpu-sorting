#ifndef BITONIC_SORT_PARALLEL_KEY_ONLY_H
#define BITONIC_SORT_PARALLEL_KEY_ONLY_H

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"


class BitonicSortParallelKeyOnly : public SortParallelKeyOnly
{
private:
    std::string _sortName = "Bitonic sort parallel key only";

    template <order_t sortOrder>
    void runBitoicSortKernel(data_t *keys, uint_t tableLen);
    template <order_t sortOrder>
    void runBitonicMergeGlobalKernel(data_t *d_keys, uint_t arrayLength, uint_t phase, uint_t step);
    template <order_t sortOrder>
    void runBitoicMergeLocalKernel(data_t *d_keys, uint_t arrayLength, uint_t phase, uint_t step);
    template <order_t sortOrder>
    void bitonicSortParallelKeyOnly(data_t *d_keys, uint_t arrayLength);
    void sortPrivate();

public:
    std::string getSortName()
    {
        return this->_sortName;
    }
};

#endif
