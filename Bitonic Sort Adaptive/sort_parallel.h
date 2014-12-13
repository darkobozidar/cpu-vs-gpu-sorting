#ifndef SORT_PARALLEL_H
#define SORT_PARALLEL_H

#include "../Utils/data_types_common.h"


double sortParallel(
    data_t *h_output, data_t *d_dataTable, data_t *d_dataBuffer, interval_t *d_intervals,
    interval_t *d_intervalsBuffer, uint_t tableLen, order_t sortOrder
);

#endif
