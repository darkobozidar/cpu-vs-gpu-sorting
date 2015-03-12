#ifndef QUICKSORT_SEQUENTIAL_KEY_VALUE_H
#define QUICKSORT_SEQUENTIAL_KEY_VALUE_H

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"
#include "data_types.h"


class QuicksortSequentialKeyValue : public SortSequentialKeyValue
{
private:
    std::string sortName = "Quicksort sequential key value";

    void exchangeElemens(data_t *elem1, data_t *elem2);
    uint_t getPivotIndex(data_t *dataTable, uint_t length);
    template <order_t sortOrder>
    uint_t partitionArray(data_t *dataKeys, data_t *dataValues, uint_t length);
    template <order_t sortOrder>
    void quickSort(data_t *dataKeys, data_t *dataValues, uint_t length);
    void sortPrivate();

public:
    std::string getSortName()
    {
        return this->sortName;
    }
};

#endif
