#ifndef SORT_PARALLEL_H
#define SORT_PARALLEL_H

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"


class BitonicSortParallelKeyOnly : public SortParallelKeyOnly
{
private:
    void runBitoicSortKernel();
    void runBitonicMergeGlobalKernel(uint_t phase, uint_t step);
    void runBitoicMergeLocalKernel(uint_t phase, uint_t step);
    void sortPrivate();

    std::string sortName = "Bitonic sort parallel key only";

public:
    std::string getSortName()
    {
        return this->sortName;
    }
};

#endif
