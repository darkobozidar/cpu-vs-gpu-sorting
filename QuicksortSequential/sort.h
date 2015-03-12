#ifndef QUICKSORT_SEQUENTIAL_H
#define QUICKSORT_SEQUENTIAL_H

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"


/*
Due to (extreme =)) optimization code for key only and key-value sorts are entirely separated.
TODO once the testing is done merge common code for key only and key-value sorts.
*/
class QuicksortSequential : public SortSequential
{
private:
    std::string _sortName = "Quicksort sequential";

    // Common
    void exchangeElemens(data_t *elem0, data_t *elem1);
    uint_t getPivotIndex(data_t *h_keys, uint_t arrayLength);

    // Key only
    template <order_t sortOrder>
    uint_t partitionArray(data_t *h_keys, uint_t arrayLength);
    template <order_t sortOrder>
    void quicksortSequential(data_t *h_keys, uint_t arrayLength);
    void sortKeyOnly();

    // Key-value
    template <order_t sortOrder>
    uint_t partitionArray(data_t *h_keys, data_t *h_values, uint_t arrayLength);
    template <order_t sortOrder>
    void quicksortSequential(data_t *h_keys, data_t *h_values, uint_t arrayLength);
    void sortKeyValue();

public:
    std::string getSortName()
    {
        return this->_sortName;
    }
};

#endif
