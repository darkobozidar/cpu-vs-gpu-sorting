#ifndef SORT_SEQUENTIAL_H
#define SORT_SEQUENTIAL_H


double sortSequential(
    data_t *dataInput, data_t *dataOutput, data_t *samples, uint_t *bucketSizes, uint_t *elementBuckets,
    uint_t tableLen, order_t sortOrder
);

#endif
