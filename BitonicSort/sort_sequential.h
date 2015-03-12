#ifndef BITONIC_SORT_SEQUENTIAL_KEY_ONLY_H
#define BITONIC_SORT_SEQUENTIAL_KEY_ONLY_H

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"


class BitonicSortSequentialKeyOnly : public SortSequentialKeyOnly
{
private:
    std::string sortName = "Bitonic sort sequential key only";

    template <order_t sortOrder>
    void bitonicSortSequentialKeyOnly(data_t *keys, uint_t arrayLength);
    void sortPrivate();

public:
    std::string getSortName()
    {
        return this->sortName;
    }
};

#endif
