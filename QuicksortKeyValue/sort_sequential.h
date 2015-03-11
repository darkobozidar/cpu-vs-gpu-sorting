#ifndef QUICKSORT_SEQUENTIAL_KEY_VALUE_H
#define QUICKSORT_SEQUENTIAL_KEY_VALUE_H

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"
#include "data_types.h"


class QuicksortSequentialKeyValue : public SortSequentialKeyValue
{
private:
    std::string sortName = "Quicksort sequential key value";

    void sortPrivate();

public:
    std::string getSortName()
    {
        return this->sortName;
    }
};

#endif
