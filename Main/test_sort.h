#ifndef TEST_SORT_H
#define TEST_SORT_H

#include <vector>
#include <memory>

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"


void generateStatistics(
    std::vector<SortSequential*> sorts, std::vector<data_dist_t> distributions, uint_t arrayLength,
    order_t sortOrder, uint_t testRepetitions, uint_t interval
);

#endif
