#ifndef BITONIC_SORT_MULTISTEP_PARALLEL_H
#define BITONIC_SORT_MULTISTEP_PARALLEL_H

#include "../Utils/data_types_common.h"
#include "../BitonicSort/sort_parallel.h"


/*
Due to (extreme =)) optimization code for key only and key-value sorts are entirely separated.
TODO once the testing is done merge common code for key only and key-value sorts.
*/
class BitonicSortMultistepParallel : public BitonicSortParallel
{
protected:
    std::string _sortName = "Bitonic sort multistep parallel";

    // Key only
    template <order_t sortOrder, uint_t threadsMerge>
    void runMultiStepKernel(data_t *d_keys, uint_t arrayLength, uint_t phase, uint_t step, uint_t degree);
    template <order_t sortOrder>
    void bitonicSortMultistepParallel(data_t *d_keys, uint_t arrayLength);
    void sortKeyOnly();

    // Key-value
    template <order_t sortOrder, uint_t threadsMerge>
    void runMultiStepKernel(
        data_t *d_keys, data_t *d_values, uint_t arrayLength, uint_t phase, uint_t step, uint_t degree
    );
    template <order_t sortOrder>
    void bitonicSortMultistepParallel(data_t *d_keys, data_t *d_values, uint_t arrayLength);
    void sortKeyValue();

public:
    std::string getSortName()
    {
        return this->_sortName;
    }
};

#endif
