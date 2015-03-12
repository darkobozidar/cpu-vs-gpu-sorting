#ifndef BITONIC_SORT_SEQUENTIAL_KEY_VALUE_H
#define BITONIC_SORT_SEQUENTIAL_KEY_VALUE_H

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"


class BitonicSortSequentialKeyValue : public SortSequentialKeyValue
{
private:
    std::string _sortName = "Bitonic sort sequential key value";

    template <order_t sortOrder>
    void bitonicSortSequentialKeyValue(data_t *h_keys, data_t *h_values, uint_t arrayLength);
    void sortPrivate();

public:
    std::string getSortName()
    {
        return this->_sortName;
    }
};

#endif
