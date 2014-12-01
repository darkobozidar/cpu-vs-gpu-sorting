#ifndef SORT_PARALLEL_H
#define SORT_PARALLEL_H

#include "../Utils/data_types_common.h"


double sortParallel(
    data_t *h_keys, data_t *h_values, data_t *d_keys, data_t *d_values, uint_t tableLen, order_t sortOrder
);

#endif
