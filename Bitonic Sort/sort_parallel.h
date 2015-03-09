#ifndef SORT_PARALLEL_H
#define SORT_PARALLEL_H

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"


class BitonicSortParallelKeyOnly : SortParallelKeyOnly
{
private:
    void runBitoicSortKernel();
    void runBitonicMergeGlobalKernel(uint_t phase, uint_t step);
    void runBitoicMergeLocalKernel(uint_t phase, uint_t step);
    void sortPrivate();
};

#endif
