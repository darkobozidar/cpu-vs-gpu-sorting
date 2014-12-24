#ifndef SORT_PARALLEL_H
#define SORT_PARALLEL_H

double sortParallel(
    data_t *h_output, data_t *d_dataTable, data_t *d_dataBuffer, data_t *d_samples, uint_t tableLen,
    order_t sortOrder
);

#endif
