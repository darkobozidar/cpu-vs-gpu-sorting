#ifndef SORT_PARALLEL_H
#define SORT_PARALLEL_H

#include "../Utils/data_types_common.h"


double sortParallel(data_t *h_input, data_t *h_output, data_t *d_dataTable, uint_t tableLen, order_t sortOrder);

#endif
