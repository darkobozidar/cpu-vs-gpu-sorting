#ifndef BITONIC_SORT_MULTISTEP_PARALLEL_H
#define BITONIC_SORT_MULTISTEP_PARALLEL_H

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"


/*
Due to (extreme =)) optimization code for key only and key-value sorts are entirely separated.
TODO once the testing is done merge common code for key only and key-value sorts.
*/
class BitonicSortMultistepParallel : public SortParallel
{
private:
    std::string _sortName = "Bitonic sort multistep parallel";

    // Key only
    template <order_t sortOrder>
    void runBitoicSortKernel(data_t *d_keys, uint_t arrayLength);
    template <order_t sortOrder>
    void runMultiStepKernel(data_t *d_keys, uint_t arrayLength, uint_t phase, uint_t step, uint_t degree);
    template <order_t sortOrder>
    void runBitonicMergeGlobalKernel(data_t *d_keys, uint_t arrayLength, uint_t phase);
    template <order_t sortOrder>
    void runBitoicMergeLocalKernel(data_t *d_keys, uint_t arrayLength, uint_t phase, uint_t step);
    template <order_t sortOrder>
    void bitonicSortMultistepParallel(data_t *d_keys, uint_t arrayLength);
    void sortKeyOnly();

    // Key-value
    //void sortKeyValue();

public:
    std::string getSortName()
    {
        return this->_sortName;
    }
};

#endif
