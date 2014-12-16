#ifndef SORT_PARALLEL_H
#define SORT_PARALLEL_H

#include "../Utils/data_types_common.h"


double sortParallel(
    data_t *h_keys, data_t *h_values, data_t *d_keys, data_t *d_values, data_t *d_keysBuffer,
    data_t *d_valuesBuffer, interval_t *d_intervals, interval_t *d_intervalsBuffer, uint_t tableLen,
    order_t sortOrder
);

#endif
